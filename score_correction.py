"""Validated, parameter-free correction for balanced binary class weights."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CORRECTION = ROOT / 'artifacts/score_correction.json'
LABELS = ('electronic', 'pop', 'ambient', 'rock')
METHOD = 'inverse_balanced_class_weight_log_odds'
CORRECTION_SHA256 = '6c92b416ccd3e4f30a8d211cec110a8a39bbcf74636fb3984986f6269655026e'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_correction(record, bundle, metadata_path, weights_path, plan_path):
    """Bind counts and the fixed formula to the exact released classifier."""
    required = {'format_version','method','labels','fit_count','fit_positive_counts',
                'fit_ids_sha256','fit_manifest_sha256','source_metadata_sha256',
                'source_weights_sha256','source_plan_sha256','class_weight',
                'logit_offsets','interpretation'}
    if set(record) != required or record['format_version'] != 1 or record['method'] != METHOD:
        raise ValueError('Unexpected score-correction artifact format.')
    if record['labels'] != list(LABELS) or bundle['labels'] != list(LABELS):
        raise ValueError('Score-correction labels differ from the model.')
    for key,path in [('source_metadata_sha256',metadata_path),('source_weights_sha256',weights_path),
                     ('source_plan_sha256',plan_path)]:
        if record[key] != sha256(path):
            raise ValueError('Score-correction artifact belongs to a different model or training plan.')
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    n = record['fit_count']; positives = record['fit_positive_counts']
    if (type(n) is not int or n <= 1 or not isinstance(positives,list) or len(positives)!=4
            or any(type(v) is not int or not 0<v<n for v in positives)
            or len(plan['fit_ids']) != n or len(set(plan['fit_ids'])) != n):
        raise ValueError('Invalid correction training counts.')
    if record['fit_ids_sha256'] != hashlib.sha256('\n'.join(plan['fit_ids']).encode()).hexdigest():
        raise ValueError('Correction training IDs changed.')
    if record['fit_manifest_sha256'] != plan['expanded_manifest_sha256']:
        raise ValueError('Correction training data identity changed.')
    weights = [r['config']['class_weight'] for r in bundle['selected']]
    if record['class_weight'] != weights or any(w not in (None,'balanced') for w in weights):
        raise ValueError('Unsupported or changed classifier class weights.')
    offsets = np.asarray(record['logit_offsets'],dtype=float)
    expected = np.array([np.log(pos/(n-pos)) if weight=='balanced' else 0.
                         for pos,weight in zip(positives,weights,strict=True)])
    if (offsets.shape!=(4,) or not np.isfinite(offsets).all()
            or not np.allclose(offsets,expected,rtol=0,atol=1e-15)):
        raise ValueError('Correction does not match the fixed class-weight formula.')
    return offsets


def load_correction(bundle, metadata_path, weights_path, plan_path, correction_path=CORRECTION):
    path = Path(correction_path)
    if not path.is_file():
        raise FileNotFoundError('Packaged score-correction artifact was not found.')
    if sha256(path) != CORRECTION_SHA256:
        raise ValueError('Packaged score-correction artifact checksum mismatch.')
    record = json.loads(path.read_text(encoding='utf-8'))
    validate_correction(record,bundle,metadata_path,weights_path,plan_path)
    return record


def corrected_thresholds(thresholds, offsets):
    """Map probability thresholds; retain 0/1 and disabled >1 sentinels exactly."""
    values=np.asarray(thresholds,dtype=float);offsets=np.asarray(offsets,dtype=float)
    if (values.shape!=(4,) or offsets.shape!=(4,) or not np.isfinite(values).all()
            or not np.isfinite(offsets).all() or np.any(values<0)):
        raise ValueError('Expected four finite thresholds and offsets.')
    result=values.copy()
    changed=(values>0)&(values<1)&(offsets!=0)
    z=np.log(values[changed])-np.log1p(-values[changed])+offsets[changed]
    result[changed]=np.exp(-np.logaddexp(0.,-z))
    return result
