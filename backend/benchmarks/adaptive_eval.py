"""Paired outer-holdout evaluation; selection happens strictly inside training."""
import numpy as np
import pandas as pd

from backend.app.services.phase3_normalizer import classify
from backend.app.services.phase4_forecaster import AdaptiveForecastEngine, Forecast, metrics


def adaptive_experiment(grid, foundation, seed=42):
    split=int(.8*len(grid))
    horizon=len(grid)-split
    histories=[grid[c].values[:split] for c in grid]
    adaptive=AdaptiveForecastEngine(foundation).batch(histories,horizon)
    fixed=foundation.batch(histories,horizon)
    rows=[]
    selection=[]
    for sku,train,a,f in zip(grid,histories,adaptive,fixed):
        actual=grid[sku].values[split:]
        regime=classify(train)['regime']
        for label,prediction in [('Adaptive',a),(f.model,f)]:
            row={'part_id':sku,'regime':regime,'method':label,'selected_model':prediction.model,
                 **metrics(train,actual,prediction),
                 'q90_coverage':float(np.mean(actual<=prediction.q(.9))),
                 'q10_q90_coverage':float(np.mean((actual>=prediction.q(.1))&(actual<=prediction.q(.9))))}
            rows.append(row)
        selection.append({'part_id':sku,**a.selection})
    raw=pd.DataFrame(rows)
    columns=['MAE','RMSE','MASE','RMSSE','Pinball_Loss_q90','q90_coverage','q10_q90_coverage']
    summary=raw.groupby(['regime','method'])[columns].mean().reset_index()
    rng=np.random.default_rng(seed)
    paired=[]
    for regime,group in raw.groupby('regime'):
        for metric in ['MASE','Pinball_Loss_q90']:
            pivot=group.pivot(index='part_id',columns='method',values=metric).dropna()
            differences=(pivot['Adaptive']-pivot[fixed[0].model]).values
            if len(differences):
                samples=rng.choice(differences,size=(2000,len(differences)),replace=True).mean(axis=1)
                paired.append({'regime':regime,'metric':metric,'n_skus':len(differences),
                    'mean_adaptive_minus_fixed':differences.mean(),'bootstrap_lower95':np.quantile(samples,.025),
                    'bootstrap_upper95':np.quantile(samples,.975),'resampling_unit':'SKU; one dataset, one outer origin'})
    return summary,raw,pd.DataFrame(paired),selection
