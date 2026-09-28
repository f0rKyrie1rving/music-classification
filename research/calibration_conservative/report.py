"""Report all prespecified results without choosing another policy after scoring."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from research.calibration.common import LABELS,read
from research.calibration.report import plt,plots
from .run import verify,verify_selection


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def create(out,destination):
    config,_,_,_=verify(out);verify_selection(out)
    destination.mkdir(parents=True);figures=destination/'figures';figures.mkdir();results=destination/'results';results.mkdir()
    for folder,names in [('evaluation',['summary.json','summary.csv']),('selection',['strengths.json','strengths.csv','candidates.json','candidates.csv']),('bootstrap',['intervals.json'])]:
        for name in names:shutil.copy2(out/folder/name,results/name)
    for name in ['config.json','freeze.json','verification.json']:shutil.copy2(out/name,results/name)
    rows=read(out/'evaluation/summary.json');strengths=read(out/'selection/strengths.json')
    for seed in config['outer_seeds']:
        for suffix in ['metrics.json','predictions.npz']:shutil.copy2(out/'evaluation'/f'{seed}_{suffix}',results/f'{seed}_{suffix}')
        shutil.copy2(out/'selection'/f'{seed}_policy.json',results/f'{seed}_policy.json')
    intervals=read(out/'bootstrap/intervals.json')
    def result(seed,method):return next(r for r in rows if r['seed']==seed and r['method']==method)
    body=[]
    for seed in config['outer_seeds']:
        raw=result(seed,'identity');full=result(seed,'sigmoid');new=result(seed,'conservative_sigmoid')
        body.append([seed,f"{raw['macro_brier']:.6f}",f"{full['macro_brier']:.6f}",f"{new['macro_brier']:.6f}",
            f"{new['delta_brier_vs_raw']:+.6f}",f"{new['delta_brier_vs_full_sigmoid']:+.6f}"])
    score_table=table(['Split','Raw Brier','Full sigmoid Brier','Conservative Brier','Conservative − raw','Conservative − full'],body)
    strength_table=table(['Split']+LABELS,[[seed]+[f"{next(r['selected_lambda'] for r in strengths if r['seed']==seed and r['label']==l):.0%}" for l in LABELS] for seed in config['outer_seeds']])
    ci_table=table(['Reference','Metric','Conservative difference','Conditional 95% interval'],
        [[name,metric,f"{item['delta']:+.6f}",f"[{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}]"]
         for name,summary in intervals.items() for metric in ['brier','log_loss']
         for item in [summary['results']['conservative_sigmoid'][metric]['macro']]])
    matrix=np.array([[next(r['selected_lambda'] for r in strengths if r['seed']==s and r['label']==l) for l in LABELS] for s in config['outer_seeds']])
    fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(4),LABELS);ax.set_yticks(range(5),config['outer_seeds'])
    for i in range(5):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:.0%}',ha='center',va='center',color='white' if matrix[i,j]>.65 else 'black',fontsize=12)
    ax.set_title('Adjustment strengths selected using calibration data only\n0% = original probability; 100% = full sigmoid correction')
    fig.colorbar(im,ax=ax,label='Selected fraction of correction',ticks=[0,.25,.5,.75,1])
    for ext in ['png','pdf']:fig.savefig(figures/f'selected_strengths.{ext}',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');x=np.arange(5);width=.25
    for k,(method,title,color) in enumerate([('sigmoid','Full sigmoid','#bb6528'),('temperature','Temperature','#8a8a8a'),('conservative_sigmoid','Conservative sigmoid','#287daf')]):
        ax.bar(x+(k-1)*width,[result(s,method)['delta_brier_vs_raw'] for s in config['outer_seeds']],width,label=title,color=color)
    ax.axhline(0,color='black',lw=1);ax.set_xticks(x,config['outer_seeds']);ax.set(xlabel='Original split',ylabel='Macro Brier change from raw',title='Fixed evaluation comparisons across all five splits\nBelow zero is better; splits are exploratory and overlap')
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
    for ext in ['png','pdf']:fig.savefig(figures/f'evaluation_changes.{ext}',dpi=160)
    plt.close(fig)
    primary_dir=figures/'primary';primary_dir.mkdir()
    with np.load(out/'evaluation'/f"{config['primary_seed']}_predictions.npz",allow_pickle=False) as data:
        plots(primary_dir,data['evaluation_y'],{m:data[m] for m in ['identity','sigmoid','conservative_sigmoid']},'Primary split: fixed raw, full and conservative probabilities')
    old_primary=read(out/'evaluation'/f"{config['primary_seed']}_metrics.json")
    label_table=table(['Label','Raw Brier','Full sigmoid Brier','Conservative Brier'],[[l]+[f"{old_primary[m]['per_label'][l]['brier']:.6f}" for m in ['identity','sigmoid','conservative_sigmoid']] for l in LABELS])
    below_raw=sum(result(s,'conservative_sigmoid')['delta_brier_vs_raw']<0 for s in config['outer_seeds'])
    below_full=sum(result(s,'conservative_sigmoid')['delta_brier_vs_full_sigmoid']<0 for s in config['outer_seeds'])
    fallback=[r for r in strengths if r['fallback_reason']];zero=int((matrix==0).sum())
    text=f'''# Conservative calibration using calibration-only internal validation

Exploratory run `{out.name}`. This policy was designed after the original evaluation
results and failure diagnostics had been inspected. It is not independent confirmation.

## Findings

The conservative policy had lower macro Brier than raw probabilities in {below_raw}/5
original splits and lower macro Brier than full sigmoid in {below_full}/5 splits.
It retained raw probabilities for {zero}/20 label/split combinations. There were
{len(fallback)} explicit identity fallbacks caused by unusable fits. Numerical details
are retained below, including every unfavorable outcome. Frequencies across these
overlapping splits are not estimates of future success probability.

{score_table}

![Evaluation comparisons](figures/evaluation_changes.png)

## How the policy was chosen

Within each original calibration pool, 94 artists were assigned to five fixed folds.
Sigmoid mappings fitted on four folds predicted the fifth; no artist crossed the
inner fitting/validation boundary. The original music classifier remained fixed and
had never fitted these calibration rows. Each row received one OOF sigmoid probability.

For each label, candidate probabilities blended raw and OOF sigmoid outputs using
0%, 25%, 50%, 75% or 100% of the correction. The lowest OOF Brier candidate was found,
then the smallest adjustment within a paired artist-cluster SE proxy of that loss was
selected. The proxy uses candidate-minus-raw losses. Its exact formula and all numerical
ties are specified in the [protocol](../../research/calibration_conservative/protocol.md).
This is a paired-SE shrinkage heuristic, not a formal significance test or a guarantee
against degradation. The SE ignores overlapping fold training and winner selection.

After selection, the sigmoid was refitted on all calibration rows and the chosen blend
was applied to the original evaluation logits. The final sigmoid predictions reproduced
the previous full sigmoid baseline. All 20 strengths were saved before any new evaluation
metric was calculated. Evaluation labels and logits are not arguments of the selection
function. The evaluation results were not used to revise the rule.

{strength_table}

![Selected strengths](figures/selected_strengths.png)

The percentages describe a fraction of the probability correction, not confidence,
accuracy or the fraction of songs classified correctly. At 0%, output equals raw
probability exactly. All four music labels retain separate binary probabilities.
Identity fallbacks, if any, are policy decisions with logged causes; no fold seeds
were retried to obtain a better selection.

## Primary split and uncertainty

{label_table}

![Primary reliability curves](figures/primary/reliability_5.png)

{ci_table}

These pointwise percentile intervals use 2,000 paired resamples of evaluation artists.
Both comparisons reuse the same resampling indices. Intervals condition on the fitted
classifier, calibration data and selected policy. They do not include method-development
history, selection/refitting uncertainty or multiple-comparison adjustment. The selected
OOF losses are tuning diagnostics, not unbiased test results. Log loss, AP, exact bin
counts and ten-bin sensitivity figures are also retained.

## Interpretation limits

Conservatism can discard a useful correction or retain a harmful one. OOF sigmoids use
less calibration data than the full-pool refit; a selected blend need not transfer
perfectly to that refit. Reusing the previous evaluation pools keeps this study
exploratory even though the new selection code excludes their labels. The five splits
overlap and no ordinary independent-sample t-test is justified.

Target enrichment, observed-tag incompleteness, first-30-second/whole-track mismatch,
prior model selection and unaudited encoder-pretraining overlap remain limitations.
No original application model was replaced and no claim of novel methodology or
publication is made. Stronger validation would require a separately fixed protocol
and previously unused songs and artists.

## Method sources and verification

The preference for a simpler option near the validation minimum follows the
[one-standard-error principle described in Stanford STATS 202](https://web.stanford.edu/class/stats202/notes/Resampling/Kfold-CV.html).
Our paired artist-cluster tolerance is a specified adaptation, not the textbook
foldwise calculation. [scikit-learn's grouped-CV guidance](https://scikit-learn.org/stable/modules/cross_validation.html)
motivates preserving groups across validation boundaries. [Bates, Hastie and Tibshirani](https://arxiv.org/abs/2104.00673)
discuss why dependencies make ordinary CV uncertainty estimates problematic; their
work does not validate this particular heuristic.

See [verification](results/verification.json), [reproduction commands](../../research/calibration_conservative/README.md),
[selected strengths](results/strengths.csv), [candidate scores](results/candidates.csv),
and [all macro results](results/summary.csv). Complete OOF predictions, inner-fold
parameters, source snapshots and bootstrap draws are in `outputs/calibration_conservative/{out.name}/`.
Tests are executed separately from numerical verification. Codex assisted with design,
implementation, execution and writing, with independent code/design checks.
'''
    (destination/'REPORT.md').write_text(text);print('Report generated:',destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();create(args.out.resolve(),args.destination.resolve())
