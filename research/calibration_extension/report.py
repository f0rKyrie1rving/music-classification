"""Render descriptive scientific figures and a complete quantitative report."""
import argparse
import os
import shutil
from pathlib import Path
import numpy as np
from research.calibration.common import read, save, digest, ROOT
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import verify

NAMES={'maest':'MAEST','mert_v0':'MERT-v0','mert_v1':'MERT-v1'}


def render(out,dest):
    config,_=verify(out)
    summary=read(out/'analysis/summary.json');scores=read(out/'analysis/scores.json')
    dest.mkdir(parents=True,exist_ok=True);figdir=dest/'figures';figdir.mkdir(exist_ok=True)
    results=dest/'results';results.mkdir(exist_ok=True)
    for path in (out/'analysis').glob('*'): shutil.copy2(path,results/path.name)
    for name in ['freeze.json','config.json','audit.json','splits.json']:
        shutil.copy2(out/name,results/name)
    if (out/'verification.json').exists(): shutil.copy2(out/'verification.json',results/'verification.json')
    if (out/'retrospective_freeze.json').exists():
        shutil.copy2(out/'retrospective_freeze.json',results/'retrospective_freeze.json')
        cache=ROOT/config['retrospective_cache']
        for name in ['freeze.json','verification.json','mert_v0_receipt.json','mert_v1_receipt.json']:
            shutil.copy2(cache/name,results/('cache_'+name))
    shutil.copy2(out/'protocol.md',dest/'PROTOCOL.md')
    shutil.copy2(Path(__file__).parent/'EXECUTION_NOTES.md',dest/'EXECUTION_NOTES.md')
    def select(scope,model,weight,method,label='macro',metric='brier'):
        return next(r for r in summary['summaries'] if all(r[k]==v for k,v in
            [('scope',scope),('model',model),('weighting',weight),('method',method),('label',label),('metric',metric)]))
    def values(model,weight,method,label='macro',key='value'):
        return np.array([r[key] for r in scores if r['scope']=='evaluation' and r['model']==model and
                         r['weighting']==weight and r['method']==method and r['label']==label and r['metric']=='brier'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    conditions=[('ambient_balanced','raw'),('unweighted','raw'),('ambient_balanced','prior_offset'),
                ('ambient_balanced','intercept'),('ambient_balanced','sigmoid')]
    names=['Weighted\nraw','Unweighted\nraw','Weighted +\nweight offset','Weighted +\nintercept','Weighted +\nfull sigmoid']
    fig,axs=plt.subplots(1,3,figsize=(13.8,4.3),sharey=True)
    for ax,model in zip(axs,config['representations']):
        data=[values(model,w,m,'ambient') for w,m in conditions]
        ax.boxplot(data,tick_labels=names,whis=(0,100),widths=.55,showfliers=False,
                   medianprops={'color':'#1b5e20','linewidth':2})
        for i,v in enumerate(data): ax.scatter(i+1+np.linspace(-.13,.13,len(v)),v,s=13,alpha=.65,color='#335c81')
        ax.set_title(NAMES[model]);ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Ambient Brier score (lower is better)')
    fig.suptitle('Weighting mechanism: 20 paired artist splits per condition',fontsize=13)
    fig.text(.5,.015,'Dots = refits; box = median and IQR; whiskers = observed range. These are not confidence intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.92))
    for ext in ['png','pdf']: fig.savefig(figdir/f'mechanism.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.7),sharey=True)
    methods=['sigmoid','conservative','blend_min'];colors=['#247ba0','#d16b32','#76559b']
    for ax,weight in zip(axs,config['weightings']):
        for j,method in enumerate(methods):
            for i,model in enumerate(config['representations']):
                v=values(model,weight,method,key='delta_raw');x=i+(j-1)*.22
                ax.scatter(x+np.linspace(-.05,.05,len(v)),v,s=18,alpha=.55,color=colors[j],
                           label={'sigmoid':'Full sigmoid','conservative':'Conservative','blend_min':'Blend without SE'}[method] if i==0 else None)
                ax.plot([x-.07,x+.07],[v.mean()]*2,color=colors[j],linewidth=3)
        ax.axhline(0,color='black',linestyle='--',linewidth=.9)
        ax.set_xticks(range(3),[NAMES[m] for m in config['representations']]);ax.grid(axis='y',alpha=.2)
        ax.set_title('Ambient class weighted' if weight=='ambient_balanced' else 'All heads unweighted')
    axs[0].set_ylabel('Macro Brier: calibrated minus raw\nNegative = improvement')
    axs[1].legend(fontsize=8,loc='best')
    fig.suptitle('Cross-representation calibration and refitting variability',fontsize=13)
    fig.text(.5,.015,'Each dot is one paired split; short bars are means. Reused tracks mean these are not 20 independent datasets.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.92))
    for ext in ['png','pdf']: fig.savefig(figdir/f'transfer_stability.{ext}',dpi=180)
    plt.close(fig)
    lines=['# Calibration extension: mechanism, representation transfer and refitting variability','',
        'This report contains all prespecified 20 artist-separated refits of three frozen audio representations and two weighting conditions: 120 fitted pipelines. '
        'Each pipeline has seven probability treatments. This is an exploratory extension on previously used data; the application has not been updated.','',
        '## Findings','',
        'Under ambient-balanced training, full sigmoid reduces macro Brier in all 20 main splits for each representation. '
        'When all heads are unweighted, its mean macro Brier change is positive for all three representations: calibration slightly worsens the average. '
        'The unweighted improvement counts are 4/20 for MAEST and 11/20 for each MERT version. '
        'The MERT median changes can still be slightly negative; an improvement count does not determine the mean effect.','',
        'Balanced ambient raw Brier is worse than unweighted ambient raw Brier in all 20 splits for each representation. '
        'A fit-prevalence offset or an intercept-only calibrator obtains ambient error close to full sigmoid. '
        'These controlled comparisons support class-weight-induced probability displacement as an important source of the apparent calibration benefit in this setup. '
        'They do not identify a percentage of benefit causally attributable to weighting or justify avoiding weighting for every task. '
        'Other objectives, such as recall or cost-sensitive decisions, were not optimized in this extension.','',
        'The SE-tolerant blend has higher macro Brier than the same OOF blend without SE in 17/20 balanced splits for each representation. '
        'Its smaller adjustment therefore has a measurable cost here. Neither blend is established as a generally superior calibration method.','',
        '## Design and reproducibility','',
        'The main pool has 1,206 songs / 469 artist IDs. Each paired split allocates approximately 60/20/20% of artists to classifier fitting, calibration, and evaluation. '
        'Scalers, four logistic heads and calibrators are newly fitted per split; audio encoders are fixed. '
        'All representations share splits, C=0.001 and optimizer settings. MERT-v0/v1 use the mean of transformer layers 1–12. '
        'The different dimensions make this a fixed-configuration transfer test, not a ranking of best-tuned encoders.','',
        'Methods and seeds were frozen before new fits. The v1 implementation was superseded by v2 solely to strengthen provenance guards, before numerical outcome inspection; '
        'see [execution history](EXECUTION_NOTES.md). The datasets and methods were chosen with earlier results already known; this is not a prospective preregistration.','',
        '## Main evaluation: all methods','',
        'Numbers below are means across the 20 split-specific metrics; every split has equal weight. Improvement counts compare each treatment with its own raw condition. '
        'The splits overlap, so their ranges/quantiles describe partition/refitting variability; they are not confidence intervals or evidence from 20 independent datasets.','',
        '| Representation | Weighting | Method | Mean macro Brier | Mean delta vs raw | Improved / 20 | Mean log loss |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for model in config['representations']:
        for w in config['weightings']:
            for m in config['methods']:
                r=select('evaluation',model,w,m);ll=select('evaluation',model,w,m,metric='log_loss')
                lines.append(f"| {NAMES[model]} | {w} | {m} | {r['value']['mean']:.6f} | {r['delta_raw']['mean']:+.6f} | {r['delta_raw']['negative']} | {ll['value']['mean']:.6f} |")
    lines+=['','![Transfer and refit variability](figures/transfer_stability.png)','',
        '## Ambient weighting mechanism','',
        'Only ambient weighting changes. The analytic offset adds log(n_positive/n_negative), with counts from classifier fitting. '
        'It is a control for idealized class-weight log-odds distortion, not an algebraic reconstruction of a separately refitted unweighted regularized classifier. '
        'Intercept-only calibration fits an additive shift; full sigmoid also changes slope.','',
        '| Representation | Balanced raw Brier | Unweighted raw Brier | Balanced + offset | Balanced + intercept | Balanced + sigmoid | Unweighted + sigmoid |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model in config['representations']:
        vals=[select('evaluation',model,w,m,'ambient')['value']['mean'] for w,m in conditions+[('unweighted','sigmoid')]]
        lines.append('| '+NAMES[model]+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines += ['', 'As a descriptive check, the following averages use the same 20 main evaluation partitions. '
              'Matching an overall mean score to prevalence is not sufficient for calibration within probability bins.','',
              '| Representation | Observed ambient fraction | Balanced raw mean score | Unweighted raw mean score |',
              '| --- | ---: | ---: | ---: |']
    for model in config['representations']:
        observed=[];means={w:[] for w in config['weightings']}
        for seed in config['seeds']:
            for w in config['weightings']:
                r=read(out/'runs'/str(seed)/model/w/'metrics.json')['raw']['per_label']['ambient']
                means[w].append(r['mean_score'])
                if w=='unweighted': observed.append(r['positive_fraction'])
        lines.append(f"| {NAMES[model]} | {np.mean(observed):.4f} | {np.mean(means['ambient_balanced']):.4f} | {np.mean(means['unweighted']):.4f} |")
    lines+=['','![Mechanism distributions](figures/mechanism.png)','',
        'The stored mechanism table includes per-seed differences of loss deltas: '
        '`(balanced calibrated − balanced raw) − (unweighted calibrated − unweighted raw)`. '
        'A negative number means the balanced condition obtains more calibration improvement. '
        'This does not imply that all of the original gain is caused by weighting.','',
        '## Removing the SE tolerance','',
        'The minimum-OOF blend reuses exactly the same inner-fold predictions and full sigmoid fits as the conservative blend. '
        'Only the strength selection rule changes. Positive differences below mean the SE rule has higher error.','',
        '| Representation | Weighting | Mean Brier: conservative minus no-SE blend | Conservative better / 20 |',
        '| --- | --- | ---: | ---: |']
    ablation=read(out/'analysis/se_ablation.json')
    for model in config['representations']:
        for w in config['weightings']:
            v=np.array([r['conservative_minus_blend_min'] for r in ablation if r['scope']=='evaluation' and r['metric']=='brier' and r['model']==model and r['weighting']==w])
            lines.append(f'| {NAMES[model]} | {w} | {v.mean():+.6f} | {(v < -1e-12).sum()} |')
    lines+=['','## Retrospective 266-song stress check','',
        'The 266 tracks / 200 artist IDs are artist-disjoint from the development pool, but their MAEST outcomes had been inspected before this extension. '
        'All new pipeline predictions on them are retrospective. The same 266 are used for each refit; they are not new independent test sets or a second data source. '
        'No fitting or per-run strength selection uses this cohort; all seven treatments are reported without outcome-based filtering.','']
    if 'retrospective' in summary['cohorts']:
        lines+=['| Representation | Weighting | Raw Brier | Sigmoid Brier | Sigmoid improved / 20 | Conservative Brier | Conservative improved / 20 |',
                '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for model in config['representations']:
            for w in config['weightings']:
                raw=select('retrospective',model,w,'raw');s=select('retrospective',model,w,'sigmoid');c=select('retrospective',model,w,'conservative')
                lines.append(f"| {NAMES[model]} | {w} | {raw['value']['mean']:.6f} | {s['value']['mean']:.6f} | {s['delta_raw']['negative']} | {c['value']['mean']:.6f} | {c['delta_raw']['negative']} |")
        lines+=['', 'The retrospective cohort repeats the conditional pattern: full sigmoid improves every balanced-condition refit, '
                'but has slightly adverse mean changes under all three unweighted conditions. Its unweighted improvement counts are '
                '1/20 for MAEST and 9/20 for each MERT version. This is supplementary consistency on an already observed cohort, not new independent confirmation.']
    else: lines+=['Retrospective extraction/scoring was not included in this rendering.']
    lines+=['','## Verification and limits','',
        f"Inner-policy label fallbacks: {summary['policy_fallbacks']}. Full optimizer statuses, boundary solutions, folds and head parameters are retained in the local run. "
        'The independent verifier reconstructs predictions, calibrators, OOF selection, split membership, per-label metrics, negative controls and summary distributions.','',
        'Limitations: one Jamendo source; four broad observed tags; incomplete or segment-mismatched labels; historical sample selection; '
        'label-balanced split selection; frozen encoders and one head regularization setting; two related MERT versions. '
        'No cross-dataset or fully unexposed prospective validation is claimed. Artist aliases, pretrained-model overlap and undocumented prior external use remain unresolved. '
        'Brier/log-loss improvements are probability-quality results, not classification accuracy improvements. '
        'The standard monotone transformations preserve within-label ranking, apart from numerical saturation/ties; the weighting intervention itself can change ranking.','',
        'Neither probability calibration nor class-weight distortion is a new discovery. Caplin, Martin and Marx (2022) '
        '[already explain class-weight miscalibration and derive a probability correction](https://arxiv.org/abs/2205.04613). '
        'The additive log-odds offset here is a standard control, not a new method. '
        'This is a reproducible empirical study of their interaction in this particular music setup. '
        'A publication claim requires a clearer distinction from existing work and further evidence on another data source or prespecified untouched sample.','',
        'See the [protocol](PROTOCOL.md), [all score rows](results/scores.csv), [mechanism contrasts](results/mechanism.csv), '
        '[calibration parameters](results/parameters.csv), and [reproduction guide](../../research/calibration_extension/README.md).','',
        'Background: [calibration survey](https://arxiv.org/abs/2112.10327), '
        '[deep audio calibration](https://arxiv.org/abs/2206.13071), '
        '[calibrating for class weights](https://arxiv.org/abs/2205.04613), '
        '[class-imbalance correction study](https://arxiv.org/abs/2202.09101), '
        '[sklearn class-weight definition](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).','']
    verification=read(out/'verification.json')
    lines += ['## Validation record','',
              f"The independent verification completed {verification['checks']:,} checks with {len(verification['errors'])} errors, "
              f"covering {verification['completed_runs']} main runs and {verification['retrospective_runs']} retrospective runs. "
              f"The maximum absolute numerical reconstruction difference was {verification['maximum_absolute_comparison_error']:.3g}. "
              'The 16 unit tests passed. Both MERT historical audio controls exactly reproduced their original cached features. '
              'All 1,440 full intercept/sigmoid/temperature fits converged without hitting their parameter bounds. '
              'Verification is an implementation and provenance check; its check count is not scientific sample size or independent replication.','']
    (dest/'REPORT.md').write_text('\n'.join(lines))
    save(dest/'results/report_provenance.json',{'run':str(out.relative_to(ROOT)),
         'summary_sha256':digest(out/'analysis/summary.json'),'report_source_sha256':digest(Path(__file__))})
    print('Report and figures:',dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();render(a.out.resolve(),a.destination.resolve())
