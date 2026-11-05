import math
import numpy as np


def replenish(forecast, stock, lead_time_days, review_days=1, on_order=0):
    length = int(lead_time_days) + review_days
    if length > len(forecast.mean):
        raise ValueError('Forecast must cover lead time plus review period')
    lead_mean = float(forecast.mean[:lead_time_days].sum())
    lead_upper = float(forecast.q(.9)[:lead_time_days].sum())
    target = float(forecast.q(.9)[:length].sum())
    return {'safety_stock': max(0., lead_upper - lead_mean), 'reorder_point': lead_upper,
            'order_up_to': target, 'recommended_order': max(0, math.ceil(target - stock - on_order)),
            'status': 'stockout_risk' if stock < lead_mean else 'overstock' if stock > 2 * target else 'healthy',
            'quantile_assumption': 'sum of marginal q90; comonotonic approximation, not calibrated lead-demand q90'}


def simulate_inventory(demand, targets, lead_time_days, initial_stock):
    """Daily lost-sales system: receive, serve demand, order for t+L. Daily review.

    targets are fixed-origin forecast-derived order-up-to levels; no future actuals.
    All policies receive the same initial stock and no opening pipeline.
    """
    demand, targets = np.asarray(demand, float), np.asarray(targets, float)
    if len(demand) != len(targets) or lead_time_days < 1 or initial_stock < 0:
        raise ValueError('Invalid simulation inputs')
    if not np.isfinite(demand).all() or not np.isfinite(targets).all() or (demand < 0).any() or (targets < 0).any():
        raise ValueError('Demand and targets must be finite and nonnegative')
    pipeline = np.zeros(len(demand) + lead_time_days + 1)
    stock, served, lost, holding = float(initial_stock), 0., 0., []
    outstanding = 0.
    for t, value in enumerate(demand):
        stock += pipeline[t]
        outstanding -= pipeline[t]
        filled = min(stock, value)
        stock -= filled
        served += filled
        lost += value - filled
        holding.append(stock)
        order = max(0, math.ceil(targets[t] - stock - outstanding))
        pipeline[t + lead_time_days] += order
        outstanding += order
    return {'Fill_Rate_pct': 100 * served / demand.sum() if demand.sum() > 0 else np.nan,
            'Average_Stock_Holding': float(np.mean(holding)), 'Lost_Units': lost,
            'Demand_Units': float(demand.sum()), 'Served_Units': served}


def fixed_origin_targets(values, test_length, lead):
    values = np.asarray(values)
    if len(values) < test_length + lead + 1:
        raise ValueError('Forecast does not cover final protection period')
    # Orders are placed after today's demand, so protection starts tomorrow.
    return np.array([values[t + 1:t + lead + 2].sum() for t in range(test_length)])
