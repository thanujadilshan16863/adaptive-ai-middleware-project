"""Exercise quantile extraction without downloading weights; not a model accuracy test."""
import contextlib
import sys
import types

import numpy as np
import pytest

from backend.app.services.phase4_forecaster import ForecastEngine


def test_chronos_quantiles_and_mean_are_not_median_alias(monkeypatch):
    fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext, float32=np.float32,
                                      tensor=lambda y, dtype: np.asarray(y, dtype=dtype))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    class Tensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Pipeline:
        def predict_quantiles(self, context, prediction_length, quantile_levels, limit_prediction_length):
            assert len(quantile_levels) == 9
            assert limit_prediction_length is False
            values = np.array([0., 0., 0., 0., 0., 1., 2., 5., 20.])
            return Tensor(np.tile(values, (len(context), prediction_length, 1))), None

    engine = ForecastEngine('chronos')
    engine.pipeline = Pipeline()
    forecasts = engine.batch([np.ones(584), np.ones(584)], 146)
    assert len(forecasts) == 2
    assert forecasts[0].quantiles.shape == (146, 4)
    assert forecasts[0].q(.5)[0] == 0
    assert forecasts[0].q(.9)[0] == 20
    assert forecasts[0].mean[0] == pytest.approx(3.8)
    assert forecasts[0].model == 'Chronos-Bolt'
