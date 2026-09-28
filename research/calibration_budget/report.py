"""Render the prespecified summaries; do not choose a method or a budget."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from research.calibration.common import LABELS,read,save
from research.calibration.report import plt
from .run import verify


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def curve(ax,summary,seed,label,config):
    ax.axhline(0,color='0.4',ls='--',lw=1)
    for method,color in [('sigmoid','#b45221'),('temperature','#2776ac')]:
        rows = sorted([r for r in summary if (r['outer_seed'],r['label'],r['method']) == (seed,label,method)],
                      key=lambda r:r['artists'])
        x = [r['artists'] for r in rows]; med = [r['delta_brier_median'] for r in rows]
        ax.plot(x,med,'o-',label=method,color=color,lw=1.7,ms=4)
        spread = [r for r in rows if r['fraction'] < 1]
        ax.fill_between([r['artists'] for r in spread],[r['delta_brier_p10'] for r in spread],
                        [r['delta_brier_p90'] for r in spread],alpha=.18,color=color)
    ax.set_xticks([24,47,71,94]); ax.grid(alpha=.2)
    ax.set_xlabel('Calibration artists'); ax.set_ylabel('Brier change from raw')


def create(out,destination):
    config,plans,_ = verify(out)
    summary = read(out/'scores/summary.json'); paired = read(out/'scores/matched_budget_comparisons.json')
    destination.mkdir(parents=True)  # No overwriting a prior report.
    figures = destination/'figures'; figures.mkdir()
    results = destination/'results'; results.mkdir()
    for name in ['summary.csv','summary.json','metrics.csv','matched_budget_comparisons.csv','status.json']:
        shutil.copy2(out/'scores'/name,results/name)
    for name in ['freeze.json','config.json','verification.json']:
        shutil.copy2(out/name,results/name)
    fig,axes = plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,seed in zip(axes.flat,config['outer_seeds']):
        curve(ax,summary,seed,'macro',config)
        ax.set_title(f"Split {seed}"+(' (primary)' if seed==config['primary_outer_seed'] else ''))
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.03,.85,'Below zero: better than raw probabilities\n\nLines: median across calibration subsets\nShading: 10th–90th percentile subset spread\n(not confidence intervals)\n\n24 / 47 / 71 artists: 30 nested repeats\n94 artists: one shared full-pool fit\n\nModels and evaluation tracks stay fixed\nwithin each split.',va='top',fontsize=11,linespacing=1.5)
    fig.suptitle('Does more calibration data help?\nExploratory reuse of five previously evaluated artist splits',fontsize=15)
    for ext in ['png','pdf']: fig.savefig(figures/f'macro_budget_curves.{ext}',dpi=150)
    plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,label in zip(axes.flat,LABELS):
        curve(ax,summary,config['primary_outer_seed'],label,config); ax.set_title(label); ax.legend(fontsize=8)
    fig.suptitle('Primary split: label-specific budget sensitivity\nShading is subset variability, not a confidence interval; lower is better',fontsize=14)
    for ext in ['png','pdf']: fig.savefig(figures/f'primary_label_curves.{ext}',dpi=150)
    plt.close(fig)
    primary = [r for r in summary if r['outer_seed']==config['primary_outer_seed'] and r['label']=='macro']
    rows = []
    for r in primary:
        rows.append([r['method'],r['artists'],f"{r['tracks_median']:g} [{r['tracks_min']}, {r['tracks_max']}]",
            f"{r['delta_brier_median']:+.6f}",
            'One fit' if r['attempted']==1 else f"[{r['delta_brier_p10']:+.6f}, {r['delta_brier_p90']:+.6f}]",
            f"{r['delta_brier_improved_count']}/{r['attempted']}",f"{r['successful']}/{r['attempted']}"])
    primary_table = table(['Method','Artists','Tracks: median [range]','Median Brier change','10–90% subset range','Better than raw','Completed'],rows)
    outer_rows = []
    for seed in config['outer_seeds']:
        for method in config['methods']:
            small = next(r for r in summary if (r['outer_seed'],r['method'],r['label'],r['fraction'])==(seed,method,'macro',.25))
            full = next(r for r in summary if (r['outer_seed'],r['method'],r['label'],r['fraction'])==(seed,method,'macro',1.))
            matches = [r for r in paired if (r['outer_seed'],r['method'],r['label'],r['low_fraction'],r['high_fraction'])==(seed,method,'macro',.25,1.) and r['success']]
            outer_rows.append([seed,method,f"{small['delta_brier_median']:+.6f}",f"{full['delta_brier_median']:+.6f}",
                f"{sum(r['larger_minus_smaller_brier']<0 for r in matches)}/{len(matches)}"])
    outer_table = table(['Split','Method','24-artist median change','94-artist change','Full pool better than small subset'],outer_rows)
    status = read(out/'scores/status.json')
    optimizer = []
    for seed in config['outer_seeds']:
        for subset in read(out/'fits'/f'{seed}_parameters.json'):
            for method,labels in subset['parameters'].items():
                for label,param in labels.items():
                    if not param['success'] or param.get('bound_hit'):
                        optimizer.append({'outer_seed':seed,'subset':subset['key'],'method':method,'label':label,**param})
    save(results/'optimizer_exceptions.json',optimizer)
    text = f'''# Calibration data budget: exploratory sensitivity study

Run: `{out.name}`. This extension asks whether probability calibration becomes more useful
or less sensitive to sample choice when more calibration artists are available.
The previous five-split results were known before this question was tested.

## Design

Keep each previous classifier, scaler and evaluation set fixed. Sample 24, 47 and 71
of the 94 calibration artists using 30 seeded permutations per outer split. Prefixes
are nested, and every song by each selected artist is retained. Use all 94 artists as
one full-budget endpoint. The smallest budgets were not label-balanced or redrawn.
No evaluation scores guided sampling, and evaluation labels were excluded from fitting.
All subset plans were frozen before fitting, and every fit was saved before new scoring.

There are 455 subsets and 910 method/subset comparisons. The same sigmoid and binary
temperature implementations and bounds from the [first study](../calibration/REPORT.md)
were reused. The four binary probabilities were not normalized across labels. The raw
model is a fixed reference within each split. Historical holdout songs were not added.

## Primary split

Differences are calibrated minus raw Brier. Negative values favor calibration. Song
counts vary because complete artist groups, not individual songs, were sampled.

{primary_table}

![Budget curves across five splits](figures/macro_budget_curves.png)

The bands describe variability across the 30 randomly selected calibration subsets,
conditional on the existing pool and evaluation sample. They are not confidence intervals
or estimates of population-wide success rates. There is only one full-pool fit per split;
its absence of a shaded range does not imply an absence of uncertainty.

## All five splits and matched comparisons

{outer_table}

The last column compares each 24-artist fit with the same split's one 94-artist endpoint.
Those 30 comparisons share an endpoint and evaluation rows; they are not independent
experiments. Full-pool agreement with the earlier study is a numerical control, not
additional independent evidence. Adjacent-budget paired comparisons, all label-level
metrics and binary log loss are retained in the result tables.

![Primary label-level budget curves](figures/primary_label_curves.png)

## Completeness and checks

Completed macro comparisons: {status['successful_method_subsets']}/{status['attempted_method_subsets']}.
Optimizer failure or bound-hit records: {len(optimizer)}; all are retained in
[optimizer exceptions](results/optimizer_exceptions.json). Single-class labels, if any,
are unavailable rather than replaced by raw predictions. Macro scores are unavailable
when one label fails. Bound-constrained estimates remain visible in summaries.

Tests check nested whole-artist sampling, one shared full-budget endpoint, deterministic
sampling unaffected by labels, no influence from unselected calibration observations or
evaluation logits on fitted parameters, and explicit single-class failures. The numerical
verification reconstructs all new probabilities, checks metrics against scikit-learn,
and checks full-budget equality with the prior predictions. See
[verification](results/verification.json).

## Limits

These curves reuse evaluation sets already inspected in the first study and remain
exploratory. They cover only subsets of 94 calibration artists per split, not arbitrarily
large calibration datasets. Classifier training size was held fixed. Target enrichment,
historical model selection, noisy tags, clip/whole-track mismatch and unaudited pretraining
overlap still apply. A stable adjustment can remain unhelpful on a particular evaluation
set. This experiment does not justify selecting a method, label policy or minimum sample
size for deployment from these curves alone.

## Reproduction and files

See [protocol](../../research/calibration_budget/protocol.md),
[commands](../../research/calibration_budget/README.md), [freeze record](results/freeze.json),
[summary](results/summary.csv) and [all metrics](results/metrics.csv).
Complete subset IDs, calibration counts, parameters, probabilities and source snapshots
are in `{out.relative_to(out.parents[2])}` under the repository root. The original study's
source, caches and results were kept unchanged. No new audio or model downloads were used.

Codex assisted with design implementation, execution, verification and drafting.
This report records exploratory results and does not claim independent confirmation.
'''
    (destination/'REPORT.md').write_text(text)
    print('Report and figures:',destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True); args = parser.parse_args()
    create(args.out.resolve(),args.destination.resolve())
