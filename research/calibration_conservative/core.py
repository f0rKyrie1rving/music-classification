"""Calibration-only fitting and selection. No evaluation inputs enter fit_policy."""
import numpy as np
from scipy.special import expit
from research.calibration.common import LABELS
from research.calibration.calibrators import fit as calibrate, predict


def make_folds(groups, outer_seed, config):
    groups=np.asarray(groups);unique=np.unique(groups);k=config['folds']
    if len(unique)<k or k<2: raise ValueError('Too few artists for grouped validation')
    rng=np.random.default_rng(np.random.SeedSequence([config['inner_seed'],outer_seed]))
    chunks=np.array_split(rng.permutation(unique),k);folds=[]
    for number,chunk in enumerate(chunks):
        valid=np.flatnonzero(np.isin(groups,chunk));train=np.flatnonzero(~np.isin(groups,chunk))
        folds.append({'fold':number,'train_indices':train.tolist(),'validation_indices':valid.tolist()})
    validate_folds(folds,groups)
    return folds


def validate_folds(folds,groups):
    groups=np.asarray(groups);seen=[];n=len(groups)
    for fold in folds:
        train=np.asarray(fold['train_indices'],dtype=int);valid=np.asarray(fold['validation_indices'],dtype=int)
        if not len(train) or not len(valid) or not np.array_equal(np.sort(np.r_[train,valid]),np.arange(n)):
            raise ValueError('Each fold must partition all calibration rows exactly once')
        if set(groups[train]) & set(groups[valid]): raise ValueError('Inner artist leakage')
        seen.extend(valid.tolist())
    if not np.array_equal(np.sort(seen),np.arange(n)): raise ValueError('Each row must be held out exactly once')


def paired_cluster_se(d,groups):
    d=np.asarray(d,dtype=float);groups=np.asarray(groups);unique=np.unique(groups)
    if d.ndim!=1 or len(d)!=len(groups) or len(unique)<2 or not np.isfinite(d).all():
        raise ValueError('Need finite paired differences and at least two artists')
    mean=float(d.mean());scores=np.array([(d[groups==g]-mean).sum() for g in unique])
    return float(np.sqrt(len(unique)/(len(unique)-1)*np.sum(scores**2)/len(d)**2))


def select_strength(y,raw,q_oof,groups,grid,tolerance=1e-12):
    y=np.asarray(y);raw=np.asarray(raw,dtype=float);q_oof=np.asarray(q_oof,dtype=float);grid=np.asarray(grid,dtype=float)
    if y.ndim!=1 or y.shape!=raw.shape or y.shape!=q_oof.shape or len(y)==0:
        raise ValueError('Expected aligned one-dimensional arrays')
    if not np.isin(y,[0,1]).all() or not np.isfinite(raw).all() or not np.isfinite(q_oof).all():
        raise ValueError('Invalid labels or probabilities')
    if ((raw<0)|(raw>1)|(q_oof<0)|(q_oof>1)).any():raise ValueError('Probabilities out of range')
    if grid.ndim!=1 or not np.array_equal(grid,np.unique(grid)) or grid[0]!=0 or grid[-1]!=1:
        raise ValueError('Strength grid must be sorted, unique, from zero to one')
    probabilities=(1-grid[:,None])*raw+grid[:,None]*q_oof
    loss=(probabilities-y)**2;means=loss.mean(1)
    best=int(np.flatnonzero(means<=means.min()+tolerance)[0])
    se=paired_cluster_se(loss[best]-loss[0],groups);threshold=float(means[best]+se)
    selected=int(np.flatnonzero(means<=threshold+tolerance)[0])
    return {'selected_lambda':float(grid[selected]),'best_lambda':float(grid[best]),'paired_se':se,
        'threshold':threshold,'selection_oof_brier':float(means[selected]),
        'candidates':[{'lambda':float(v),'brier':float(means[i]),'delta_brier':float(means[i]-means[0])} for i,v in enumerate(grid)],
        'interpretation':'selection heuristic; paired SE is not a confidence guarantee'}


def fit_policy(zcal,ycal,groups,folds,fit_config,config):
    zcal=np.asarray(zcal,dtype=float);ycal=np.asarray(ycal);groups=np.asarray(groups)
    if zcal.shape!=ycal.shape or zcal.ndim!=2 or zcal.shape[1]!=4 or len(groups)!=len(zcal):
        raise ValueError('Expected calibration-only N x 4 arrays')
    if not np.isfinite(zcal).all() or not np.isin(ycal,[0,1]).all():raise ValueError('Invalid calibration input')
    validate_folds(folds,groups);oof=np.full(zcal.shape,np.nan);policy={'labels':{}}
    for j,label in enumerate(LABELS):
        records=[];failure=[]
        for fold in folds:
            train=np.array(fold['train_indices']);valid=np.array(fold['validation_indices'])
            if len(np.unique(ycal[train,j]))!=2:
                record={'success':False,'reason':'single_class_inner_training'}
            else:
                record=calibrate(zcal[train,j],ycal[train,j],'sigmoid',fit_config)
                if record['success']:
                    probability=predict(zcal[valid,j],record)
                    if np.isfinite(probability).all():oof[valid,j]=probability
                    else:record={**record,'success':False,'reason':'nonfinite_inner_prediction'}
            records.append({'fold':fold['fold'],'parameters':record})
            if not record['success']:failure.append(f"inner_fold_{fold['fold']}: {record.get('reason',record.get('message'))}")
        selection=None;full=None
        if not failure:
            selection=select_strength(ycal[:,j],expit(zcal[:,j]),oof[:,j],groups,
                                      config['lambda_grid'],config['tie_tolerance'])
            full=calibrate(zcal[:,j],ycal[:,j],'sigmoid',fit_config)
            if not full['success']:failure.append('full_fit_failed: '+full.get('message',''))
            elif not np.isfinite(predict(zcal[:,j],full)).all():failure.append('nonfinite_full_prediction')
        policy['labels'][label]={'selected_lambda':0. if failure else selection['selected_lambda'],
            'selection':selection,'inner_fits':records,'full_fit':full,
            'fallback_reason':'; '.join(failure) if failure else None}
    return policy,oof


def predict_policy(zeval,policy):
    z=np.asarray(zeval,dtype=float)
    if z.ndim!=2 or z.shape[1]!=4 or not np.isfinite(z).all():raise ValueError('Expected finite N x 4 logits')
    raw=expit(z);result=raw.copy()
    for j,label in enumerate(LABELS):
        record=policy['labels'][label];strength=record['selected_lambda']
        if not 0<=strength<=1:raise ValueError('Invalid selected strength')
        if strength:
            full=record['full_fit']
            if record['fallback_reason'] or not full or not full['success']:raise ValueError('Nonzero adjustment without a valid full fit')
            q=predict(z[:,j],full);result[:,j]=(1-strength)*raw[:,j]+strength*q
    return result
