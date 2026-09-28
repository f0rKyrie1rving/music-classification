"""Render fixed diagnostic tables and scientific figures."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from research.calibration.common import read,LABELS
from research.calibration.report import plt
from .run import verify


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def create(out,destination):
    config,_=verify(out);focus=config['focus_seed'];data=out/'results'
    summaries=read(data/'probability_summary.json');sensitivity=read(data/'artist_sensitivity.json')
    decomposition=read(data/'brier_decomposition.json');conditional=read(data/'conditional_raw_scores.json')
    destination.mkdir(parents=True);figures=destination/'figures';figures.mkdir();results=destination/'results';results.mkdir()
    for path in data.iterdir():shutil.copy2(path,results/path.name)
    for name in ['freeze.json','config.json','verification.json']:shutil.copy2(out/name,results/name)

    def lookup(role,label,method='identity',seed=focus):
        return next(r for r in summaries if (r['seed'],r['role'],r['label'],r['method'])==(seed,role,label,method))

    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    x=np.arange(4);width=.25
    for ax,role in zip(axes,['calibration','evaluation']):
        rate=[lookup(role,l)['positive_fraction'] for l in LABELS]
        raw=[lookup(role,l)['mean_probability'] for l in LABELS]
        sig=[lookup(role,l,'sigmoid')['mean_probability'] for l in LABELS]
        for offset,values,label,color in [(-1,rate,'Observed label fraction','#777777'),(0,raw,'Raw mean score','#2776ac'),(1,sig,'Sigmoid mean score','#b45221')]:
            bars=ax.bar(x+offset*width,values,width,label=label,color=color)
            ax.bar_label(bars,labels=[f'{v:.1%}' for v in values],fontsize=8,padding=3)
        ax.set_xticks(x,LABELS);ax.set_ylim(0,.52);ax.set_ylabel('Fraction / mean probability')
        ax.set_title(role+(' (fitting-set diagnostic)' if role=='calibration' else ' (held out from fitting)'))
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Post-hoc case study: split {focus}\nMean agreement does not establish calibration in every score range',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(figures/f'focus_mean_scores.{ext}',dpi=160)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for ax,method in zip(axes,config['methods']):
        for j,label in enumerate(LABELS):
            values=[next(r['delta_brier'] for r in decomposition if (r['seed'],r['role'],r['method'],r['label'])==(seed,'evaluation',method,label)) for seed in config['context_seeds']]
            ax.plot(range(5),values,'o-',label=label,ms=4)
        ax.axhline(0,color='0.4',ls='--',lw=1);ax.set_xticks(range(5),[str(s)[-4:] for s in config['context_seeds']])
        ax.axvspan(.8,1.2,color='grey',alpha=.12)
        ax.set(title=method,xlabel='Split seed suffix (0927 is focus)',ylabel='Label Brier change from raw')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('All five splits retained: label-specific evaluation changes\nNegative values favor calibration; these are overlapping exploratory splits',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(figures/f'label_changes_all_splits.{ext}',dpi=160)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for m,method in enumerate(config['methods']):
        rows=[next(r for r in sensitivity if r['seed']==s and r['method']==method) for s in config['context_seeds']]
        point=np.array([r['track_weighted_delta'] for r in rows]);low=np.array([r['leave_one_artist_out_min'] for r in rows]);high=np.array([r['leave_one_artist_out_max'] for r in rows])
        ax.errorbar(np.arange(5)+(m-.5)*.12,point,yerr=np.array([point-low,high-point]),fmt='o',capsize=4,label=method)
    ax.axhline(0,color='0.4',ls='--',lw=1);ax.set_xticks(range(5),[str(s) for s in config['context_seeds']]);ax.legend()
    ax.set(xlabel='Split',ylabel='Macro Brier change from raw',title='Deleting one evaluation artist at a time\nWhiskers are min/max sensitivity, NOT confidence intervals; no refitting')
    ax.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(figures/f'artist_sensitivity.{ext}',dpi=160)
    plt.close(fig)

    composition=table(['Label','Calibration positive fraction','Evaluation positive fraction','Evaluation raw mean','Evaluation sigmoid mean'],
        [[l]+[f'{v:.2%}' for v in [lookup('calibration',l)['positive_fraction'],lookup('evaluation',l)['positive_fraction'],
           lookup('evaluation',l)['mean_probability'],lookup('evaluation',l,'sigmoid')['mean_probability']]] for l in LABELS])
    losses_table=table(['Label','Calibration Brier change (in sample)','Evaluation Brier change','Contribution to macro change'],
        [[l]+[f'{v:+.6f}' for v in [lookup('calibration',l,'sigmoid')['brier']-lookup('calibration',l)['brier'],
           lookup('evaluation',l,'sigmoid')['brier']-lookup('evaluation',l)['brier'],
           (lookup('evaluation',l,'sigmoid')['brier']-lookup('evaluation',l)['brier'])/4]] for l in LABELS])
    artist_table=table(['Split','Method','Track-weighted change','Artist-equal change','Delete-one range','Delete-one results below zero'],
        [[r['seed'],r['method'],f"{r['track_weighted_delta']:+.6f}",f"{r['artist_equal_delta']:+.6f}",
          f"[{r['leave_one_artist_out_min']:+.6f}, {r['leave_one_artist_out_max']:+.6f}]",
          f"{r['leave_one_artist_out_below_zero']}/{r['leave_one_artist_out_count']}"] for r in sensitivity])
    conditional_table=table(['Label','Observed target','Calibration raw mean','Evaluation raw mean','Calibration/evaluation tracks'],
        [[label,target,f"{next(r['raw_mean'] for r in conditional if (r['seed'],r['role'],r['label'],r['target'])==(focus,'calibration',label,target)):.4f}",
          f"{next(r['raw_mean'] for r in conditional if (r['seed'],r['role'],r['label'],r['target'])==(focus,'evaluation',label,target)):.4f}",
          '/'.join(str(next(r['tracks'] for r in conditional if (r['seed'],r['role'],r['label'],r['target'])==(focus,role,label,target))) for role in ['calibration','evaluation'])]
         for label in LABELS for target in [0,1]])
    text=f'''# Why one split worsened: post-hoc calibration diagnostics

Run `{out.name}`. Focus split: {focus}, selected after the first two studies showed
unfavorable sigmoid calibration. The analysis is descriptive and outcome-selected.
It uses saved probabilities and parameters only; **no model or calibrator was fitted**.
All five original splits remain in the tables and figures.

## Calibration and evaluation composition

{composition}

![Means and observed fractions](figures/focus_mean_scores.png)

The calibration column describes the data used to fit the correction. Its fitted mean
agreement is an in-sample diagnostic. A match between a mean score and a label fraction
does not establish agreement throughout the score range. Fixed five-bin tables retain
counts of both tracks and artists in [raw-bin diagnostics](results/fixed_raw_bins.csv).

## Where the probability loss changes

{losses_table}

Calibration Brier is reported descriptively even though the optimizer minimized log loss.
Evaluation values use the identical rows and classifier before and after recalibration.
Label contributions add to the macro change. Complete binary log-loss values are in
[probability summaries](results/probability_summary.csv).

![All split changes](figures/label_changes_all_splits.png)

For q as calibrated probability, p as raw probability and d=q-p, the exact identity is
mean[(q-y)^2-(p-y)^2] = mean[d^2] + 2*mean[d*(p-y)]. The first term is adjustment magnitude;
the second describes alignment with the raw error on these rows. This algebra does not
identify a causal mechanism. Both terms and positive/negative-target contributions are
saved for every label and subset in [decomposition](results/brier_decomposition.csv).

## Conditional score behavior

{conditional_table}

Conditioning on the observed label exposes score differences that cannot be summarized
by class proportions alone. These are descriptive sample differences; neither equality
nor a distribution-shift model is tested. Quantiles and artist counts accompany the full
[conditional table](results/conditional_raw_scores.csv).

## Artist sensitivity

{artist_table}

![Evaluation-artist deletion sensitivity](figures/artist_sensitivity.png)

The track-weighted change is the original estimand. Artist-equal weighting answers a
different question and is shown only as sensitivity. Deleting one evaluation artist
does not refit the model and is not a rule for cleaning the data. All deletion results
and signed contributions are retained in [artist contributions](results/artist_contributions.csv).
The top-five concentration statistic uses the sum of positive contributions as its
denominator, avoiding cancellation by beneficial contributions. It does not measure
the fraction of a causal error explained by those artists.

## Limits and reproducibility

The focus split was chosen after its result was known. Evaluation labels are used here
to understand an existing outcome, not to select a new method, label policy or threshold.
Mean gaps do not prove label shift, score-bin differences do not establish causality,
and a stable delete-one sign is not a significance test. This analysis addresses
evaluation artist composition, not the effect of deleting calibration artists and refitting.
All prior limitations concerning enriched sampling, historical development use, observed
tags and encoder pretraining overlap continue to apply.

See [protocol](../../research/calibration_diagnostics/protocol.md),
[commands](../../research/calibration_diagnostics/README.md),
[freeze record](results/freeze.json), [numerical verification](results/verification.json),
and [plain-language explanation](SUMMARY_ZH.md). New tests check exact loss identities,
artist aggregation, direct deletion calculations, fixed raw-bin membership and empty
cases. Full inputs and old source hashes were rechecked; the original studies remain
unchanged. Codex assisted with implementation and reporting, with independent read-only
agent checks of the design and numerical conclusions. No independent scientific
validation is claimed.
'''
    (destination/'REPORT.md').write_text(text);print('Created diagnostic report:',destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();create(args.out.resolve(),args.destination.resolve())
