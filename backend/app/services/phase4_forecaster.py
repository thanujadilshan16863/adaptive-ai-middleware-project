"""Zero-shot forecasts; baselines never train on the holdout."""
from dataclasses import dataclass
import threading

import numpy as np

from backend.app.config import settings

QUANTILES = np.array([.1, .5, .8, .9])


@dataclass
class Forecast:
    model: str
    mean: np.ndarray
    quantiles: np.ndarray
    provenance: str
    mean_method: str = 'point forecast'
    selection: dict | None = None

    def q(self, level):
        return self.quantiles[:, list(QUANTILES).index(level)]


def baseline_point(y, horizon, model, alpha=.1, beta=.1, season_length=7):
    y = np.asarray(y, dtype=float)
    if model == 'SeasonalNaive':
        return np.resize(y[-min(season_length, len(y)):], horizon)
    nonzero = np.flatnonzero(y > 0)
    if not len(nonzero):
        return np.zeros(horizon)
    first = int(nonzero[0])
    size, interval = y[first], float(first + 1)
    probability = 1. / interval
    elapsed = 1
    for value in y[first + 1:]:
        if model == 'TSB':
            probability += beta * ((value > 0) - probability)
        if value > 0:
            size += alpha * (value - size)
            interval += alpha * (elapsed - interval)
            elapsed = 1
        else:
            elapsed += 1
    point = size * probability if model == 'TSB' else size / interval
    if model == 'CrostonSBA':
        point *= 1 - alpha / 2
    if model not in ('TSB', 'CrostonClassic', 'CrostonSBA'):
        raise ValueError('Unknown baseline')
    return np.full(horizon, point)


class ForecastEngine:
    def __init__(self, mode='offline'):
        if mode not in ('offline', 'chronos'):
            raise ValueError('Use offline or chronos mode')
        self.mode, self.pipeline = mode, None
        self.lock = threading.Lock()

    def load(self):
        if self.pipeline is None:
            import torch
            from chronos import BaseChronosPipeline
            self.pipeline = BaseChronosPipeline.from_pretrained(
                settings.chronos_model, device_map=settings.chronos_device, torch_dtype=torch.float32,
                cache_dir=settings.chronos_cache_dir)

    def forecast(self, y, horizon=28):
        return self.batch([y], horizon)[0]

    def batch(self, histories, horizon=28):
        histories = [np.asarray(y, dtype=float) for y in histories]
        if not 1 <= horizon <= 366 or any(not len(y) or not np.isfinite(y).all() or (y < 0).any() for y in histories):
            raise ValueError('Invalid forecast history or horizon')
        if self.mode == 'offline':
            return [Forecast('EmpiricalOffline', np.full(horizon, y.mean()),
                             np.tile(np.quantile(y, QUANTILES), (horizon, 1)),
                             'offline empirical distribution; NOT Chronos', 'historical arithmetic mean') for y in histories]
        import torch
        with self.lock, torch.inference_mode():
            self.load()
            all_levels = np.arange(1, 10) / 10
            outputs = []
            for offset in range(0, len(histories), settings.chronos_batch_size):
                context = [torch.tensor(y, dtype=torch.float32) for y in histories[offset:offset + settings.chronos_batch_size]]
                # Positional context supports both context/inputs API parameter names.
                quantiles, _median_alias = self.pipeline.predict_quantiles(
                    context, prediction_length=horizon, quantile_levels=all_levels.tolist(),
                    limit_prediction_length=False)
                q = np.sort(np.maximum(quantiles.cpu().numpy(), 0), axis=-1)
                # Integrate piecewise-linear quantiles with constant tails at q.1/q.9.
                # Bolt's second return value is a median, not an expectation.
                mean = np.trapezoid(np.concatenate([q[..., :1], q, q[..., -1:]], axis=-1),
                                    x=np.r_[0, all_levels, 1], axis=-1) if hasattr(np, 'trapezoid') else np.trapz(
                    np.concatenate([q[..., :1], q, q[..., -1:]], axis=-1), x=np.r_[0, all_levels, 1], axis=-1)
                outputs.extend(Forecast('Chronos-Bolt', m, a[:, [0, 4, 7, 8]], settings.chronos_model,
                                        'quantile integral, constant endpoint tails; approximate mean') for m, a in zip(mean, q))
            return outputs


def baselines(y, horizon):
    # One-step expanding-origin residual calibration, using training data only.
    results = []
    for model in ['CrostonClassic', 'CrostonSBA', 'TSB', 'SeasonalNaive']:
        point = baseline_point(y, horizon, model)
        indices = range(max(7, len(y) - 100), len(y))
        residuals = [y[t] - baseline_point(y[:t], 1, model)[0] for t in indices]
        adjustment = np.quantile(residuals or [0.], QUANTILES)
        q = np.maximum(point[:, None] + adjustment[None, :], 0)
        results.append(Forecast(model, point, q, 'classical + training one-step residual quantiles'))
    return results


class AdaptiveForecastEngine:
    """Per-SKU selection on an inner temporal validation window, never test data.

    Candidate pools depend on the inner-training regime. Rank by equal-weight
    MAE and q90 pinball loss (same units); deterministic ties favor pool order.
    Refit the selected algorithm on all available history after selection.
    """
    def __init__(self, foundation):
        self.foundation = foundation
        self.mode = foundation.mode

    @property
    def pipeline(self):
        return self.foundation.pipeline

    def load(self):
        self.foundation.load()

    def forecast(self, y, horizon=28):
        return self.batch([y], horizon)[0]

    def batch(self, histories, horizon=28):
        from backend.app.services.phase3_normalizer import classify
        histories = [np.asarray(y, dtype=float) for y in histories]
        if any(len(y) < 56 for y in histories):
            raise ValueError('Adaptive selection requires at least 56 periods per SKU')
        validation_length = min(56, min(len(y) // 5 for y in histories))
        inner_train = [y[:-validation_length] for y in histories]
        foundation_validation = self.foundation.batch(inner_train, validation_length)
        selections = []
        for y, train, foundation in zip(histories, inner_train, foundation_validation):
            regime = classify(train)['regime']
            classical = {f.model: f for f in baselines(train, validation_length)}
            pools = {'Smooth': ['SeasonalNaive', foundation.model, 'CrostonSBA'],
                     'Erratic': [foundation.model, 'TSB', 'SeasonalNaive'],
                     'Intermittent': ['CrostonSBA', 'TSB', foundation.model, 'CrostonClassic'],
                     'Lumpy': [foundation.model, 'TSB', 'CrostonSBA', 'CrostonClassic']}
            pool = pools.get(regime, ['CrostonSBA', 'TSB', foundation.model])
            candidates = {**classical, foundation.model: foundation}
            scores = {}
            for name in pool:
                result = metrics(train, y[-validation_length:], candidates[name])
                scores[name] = float(.5 * result['MAE'] + .5 * result['Pinball_Loss_q90'])
            winner = min(pool, key=lambda name: scores[name])
            selections.append({'strategy': 'adaptive', 'selected_model': winner,
                'inner_training_regime': regime, 'validation_periods': validation_length,
                'selection_score': scores[winner], 'candidate_scores': scores,
                'objective': '0.5*MAE + 0.5*q90 pinball; inner temporal validation only',
                'refit': 'all available history after selection'})
        selected_indices = [i for i, s in enumerate(selections) if s['selected_model'] in ('Chronos-Bolt', 'EmpiricalOffline')]
        foundation_final = self.foundation.batch([histories[i] for i in selected_indices], horizon) if selected_indices else []
        forecasts = dict(zip(selected_indices, foundation_final))
        outputs = []
        for i, (y, selection) in enumerate(zip(histories, selections)):
            f = forecasts[i] if i in forecasts else next(f for f in baselines(y, horizon) if f.model == selection['selected_model'])
            f.selection = selection
            outputs.append(f)
        return outputs


def build_engine():
    foundation = ForecastEngine(settings.forecast_mode)
    return AdaptiveForecastEngine(foundation) if settings.forecast_strategy == 'adaptive' else foundation


def metrics(train, actual, forecast, season_length=7):
    train, actual = np.asarray(train), np.asarray(actual)
    error = actual - forecast.mean
    scale = train[season_length:] - train[:-season_length]
    mae, mse = np.mean(np.abs(error)), np.mean(error**2)
    a = np.mean(np.abs(scale)) if len(scale) else 0
    b = np.mean(scale**2) if len(scale) else 0
    e = actual - forecast.q(.9)
    return {'MAE': float(mae), 'RMSE': float(np.sqrt(mse)),
            'MASE': float(mae / a) if a > 0 else np.nan,
            'RMSSE': float(np.sqrt(mse / b)) if b > 0 else np.nan,
            'Pinball_Loss_q90': float(np.mean(np.maximum(.9 * e, -.1 * e))),
            'Pinball_Sum_q90': float(np.sum(np.maximum(.9 * e, -.1 * e))),
            'Absolute_Demand_Units': float(np.sum(np.abs(actual))),
            'Weighted_Quantile_Loss_q90': float(2 * np.sum(np.maximum(.9 * e, -.1 * e)) / np.sum(np.abs(actual))) if np.sum(np.abs(actual)) > 0 else np.nan}
