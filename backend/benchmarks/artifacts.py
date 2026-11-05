from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def export_table(frame, name, output):
    frame.to_csv(Path(output) / f'{name}.csv', index=False)
    (Path(output) / f'{name}.tex').write_text(frame.to_latex(index=False, float_format='%.4f', escape=True), encoding='utf-8')


def figures(output, grid, regimes, schema, latency, example, offline):
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.15)
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'})
    suffix = ' — offline demonstration' if offline else ''
    output = Path(output)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    zeros = grid.eq(0).mean().rename('zero_fraction').to_frame().join(regimes.set_index('part_id')['regime'])
    sns.boxplot(data=zeros, x='regime', y='zero_fraction', ax=axes[0], color='#70a7bb')
    axes[0].set(xlabel='', ylabel='Fraction of zero-demand days', title='Sparsity by training regime')
    axes[0].tick_params(axis='x', rotation=20)
    sku = example['sku']
    sparse = pd.Series(example['actual'], index=example['dates'])
    illustration_label = 'Supplemental synthetic ' if example.get('supplemental') else ''
    regime = example.get('regime', 'Lumpy')
    axes[1].plot(sparse.index, sparse, color='#287c8e', lw=.7)
    axes[1].scatter(sparse.index[sparse.eq(0)], sparse[sparse.eq(0)], color='#e3a448', s=3, label='Zero demand')
    axes[1].set(ylabel='Daily units', xlabel='Date', title=f'{illustration_label}{regime} SKU {sku}\n{sparse.eq(0).mean():.1%} zero days')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend(fontsize=8)
    save(fig, 'fig_4_1_missing_and_zeros')

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(data=regimes, x='ADI', y='CV2', hue='regime', style='regime', ax=ax, s=40)
    ax.set(xscale='log', yscale='log', xlabel='Average demand interval (ADI)', ylabel='Squared coefficient of variation (CV²)',
           title='Syntetos–Boylan demand classification (training only)')
    ax.axvline(1.32, color='black', ls='--', lw=1)
    ax.axhline(.49, color='black', ls='--', lw=1)
    ax.text(1.34, ax.get_ylim()[1] / 1.5, 'ADI = 1.32', fontsize=8)
    ax.text(ax.get_xlim()[1] / 1.8, .53, 'CV² = 0.49', fontsize=8)
    save(fig, 'fig_4_2_syntetos_boylan_quadrants')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = schema.copy()
    plotted['method'] = plotted.mapper + ' [' + plotted.backend + ']'
    sns.barplot(data=plotted, x='tier', y='F1', hue='method', ax=ax, errorbar=None)
    ax.set(ylim=(0, 1.05), xlabel='Schema difficulty tier', title='Schema alignment ablation' + suffix)
    ax.legend(title='', loc='upper center', bbox_to_anchor=(.5, -.17), fontsize=8)
    save(fig, 'fig_4_3_schema_mapping_f1')

    fig, ax = plt.subplots(figsize=(10, 4.7))
    split, dates = example['split'], example['dates']
    actual, f = example['actual'], example['foundation']
    n = len(dates) - split
    ax.plot(dates[max(0, split-60):], actual[max(0, split-60):], color='#263238', lw=1, label='Actual demand')
    ax.fill_between(dates[split:], f.q(.1)[:n], f.q(.9)[:n], color='#70b8d0', alpha=.35, label='q10–q90 marginal band')
    ax.fill_between(dates[split:], f.q(.5)[:n], f.q(.8)[:n], color='#167c9b', alpha=.25, label='q50–q80 marginal band')
    ax.plot(dates[split:], f.q(.5)[:n], color='#167c9b', label=f'{f.model} q50')
    ax.plot(dates[split:], example['sba'].mean[:n], color='#c27929', ls='--', label='SBA')
    ax.axvline(dates[split], color='black', ls=':', label='80/20 split')
    ax.set(ylabel='Daily units', xlabel='Date', title=f'{illustration_label}{regime} SKU {example["sku"]}: fixed-origin forecasts' + suffix)
    ax.legend(fontsize=8, ncol=3)
    save(fig, 'fig_4_4_forecast_comparison_lumpy_sku')

    fig, ax = plt.subplots(figsize=(8, 4.7))
    latency.pivot(index='n_skus', columns='phase', values='Latency_ms').plot.bar(stacked=True, ax=ax, colormap='viridis')
    ax.set(xlabel='SKUs', ylabel='Median execution time (ms)', title='Warm middleware phase latency' + suffix)
    ax.legend(title='', fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    save(fig, 'fig_4_5_middleware_latency_breakdown')
