"""Application v1.1: correct class-weight score bias with unchanged tag decisions."""

import argparse
import json
from pathlib import Path

import numpy as np

from maest_hf_features import FEATURE_WIDTH, HfMaestEncoder
from predict_maest import LABELS, POLICIES, METADATA, WEIGHTS, PLAN, load_bundle
from score_correction import CORRECTION, load_correction, corrected_thresholds

SCORE_MODES = ('weight_corrected', 'raw')
VERSION_FILE = Path(__file__).resolve().parent / 'app_version.txt'
APP_VERSION = VERSION_FILE.read_text(encoding='utf-8').strip() if VERSION_FILE.is_file() else 'development'


def predict_vector(bundle, feature, policy='f1', score_mode='weight_corrected', correction=None):
    """Retain v1.0 arithmetic/decisions, returning corrected and original scores."""
    if policy not in POLICIES or score_mode not in SCORE_MODES:
        raise ValueError('Unknown prediction policy or score mode.')
    feature=np.asarray(feature,dtype=float)
    if feature.shape!=(FEATURE_WIDTH,) or not np.isfinite(feature).all():
        raise ValueError(f'Expected one finite MAEST feature vector of width {FEATURE_WIDTH}.')
    z=np.sum(((feature-bundle['means'])/bundle['scales'])*bundle['coefficients'],axis=1)+bundle['intercepts']
    if not np.isfinite(z).all():
        raise ValueError('Classifier produced non-finite logits.')
    raw=np.exp(-np.logaddexp(0.,-z))
    original_thresholds=np.asarray(bundle['thresholds'][policy],dtype=float)
    if score_mode=='weight_corrected':
        if correction is None:
            raise ValueError('Validated score correction is required for corrected mode.')
        offsets=np.asarray(correction['logit_offsets'],dtype=float)
        if offsets.shape!=(4,) or not np.isfinite(offsets).all():
            raise ValueError('Invalid score correction offsets.')
        scores=raw.copy();changed=offsets!=0
        scores[changed]=np.exp(-np.logaddexp(0.,-(z[changed]+offsets[changed])))
        thresholds=corrected_thresholds(original_thresholds,offsets)
    else:
        scores=raw;thresholds=original_thresholds
    return [{'label':label,'score':round(float(scores[j]),4),
             'threshold':round(float(thresholds[j]),4),
             'selected':bool(raw[j]>=original_thresholds[j]),
             'raw_score':round(float(raw[j]),4),'raw_threshold':round(float(original_thresholds[j]),4)}
            for j,label in enumerate(LABELS)]


def classify(audio, device='cpu', policy='f1', encoder=None,
             metadata_path=METADATA, weights_path=WEIGHTS, score_mode='weight_corrected',
             correction_path=CORRECTION):
    bundle=load_bundle(metadata_path,weights_path)
    if score_mode not in SCORE_MODES: raise ValueError('Unknown score mode.')
    correction=(load_correction(bundle,metadata_path,weights_path,PLAN,correction_path)
                if score_mode=='weight_corrected' else None)
    encoder=encoder or HfMaestEncoder(device=device)
    feature=encoder.extract(audio)
    details=predict_vector(bundle,feature,policy,score_mode,correction)
    warning=('Experimental estimates: ambient has a class-weight correction; the other labels are unchanged. '
             'These are not guaranteed confidence percentages.' if score_mode=='weight_corrected'
             else 'Original classifier scores; not calibrated confidence percentages.')
    if policy=='precision_target': warning+=' Selective policy suppresses pop and ambient and has low coverage.'
    return {'audio':str(Path(audio)),'analyzed':'Exactly the first 30 seconds of an input at least 30 seconds long',
            'model':'Discogs-MAEST block 7 plus four logistic-regression heads',
            'application_version':APP_VERSION,'score_mode':score_mode,'policy':policy,
            'predicted_tags':[d['label'] for d in details if d['selected']],
            'scores':details,'warning':warning}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio',type=Path);parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--policy',choices=POLICIES,default='f1')
    parser.add_argument('--raw-scores',action='store_true',help='Restore original scores and decision thresholds.')
    args=parser.parse_args()
    result=classify(args.audio,args.device,args.policy,score_mode='raw' if args.raw_scores else 'weight_corrected')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    try: main()
    except (OSError,ValueError,RuntimeError,KeyError,IndexError) as error:
        raise SystemExit(f'Could not classify audio: {error}') from None
