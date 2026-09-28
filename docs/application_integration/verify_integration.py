"""Replay application v1.2 against all frozen validation rows, without extraction.

This integration audit uses the original 372 cached feature rows. It checks the
public application entry point, independently calculated row arithmetic and the
saved predictions for both policies, while preserving all old research files.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import application_predict
import application_release
import predict_app
from research.application_candidate_validation import pipeline


SOURCE = ROOT / 'outputs/application_candidate_validation/20260928_v1'
OUTPUT = ROOT / 'docs/application_integration/verification.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0

    def check(self, condition, message):
        self.checks += 1
        if not bool(condition):
            raise AssertionError(message)

    def exact(self, actual, expected, message):
        self.check(np.array_equal(actual, expected), message)


class CachedEncoder:
    def __init__(self, feature):
        self.feature, self.calls = feature, 0

    def extract(self, audio):
        self.calls += 1
        return self.feature


def logits(x, bundle):
    # Independent row arithmetic, avoiding calls to either prediction function.
    return np.asarray([np.sum(((np.asarray(row, dtype=np.float64) - bundle['means'])
        / bundle['scales']) * bundle['coefficients'], axis=1) + bundle['intercepts']
        for row in x], dtype=np.float64)


def verify(source=SOURCE):
    audit = Audit()
    pipeline.verify_frozen(source)
    release, candidate = application_predict.load_application()
    audit.check(application_release.APP_VERSION == '1.2.0', 'Current application version')
    audit.check(predict_app.APP_VERSION == '1.1.0', 'Frozen baseline version')
    for key, filename in [('freeze_sha256', 'freeze.json'),
                          ('summary_sha256', 'summary.json'),
                          ('verification_sha256', 'verification.json'),
                          ('independent_verification_sha256', 'independent_verification.json'),
                          ('predictions_sha256', 'predictions.npz')]:
        audit.check(sha(source / filename) == release['validation'][key], 'Release evidence: ' + filename)
    audit.check(read(source / 'summary.json')['assessment']['verdict'] == 'go', 'Accepted candidate')
    audit.check(read(source / 'verification.json')['state'] == 'passed', 'Original replay verification')
    audit.check(read(source / 'independent_verification.json')['state'] == 'passed', 'Independent validation')
    feature_record = read(source / 'features.json')
    audit.check(sha(source / 'features.npy') == feature_record['feature_sha256'], 'Unchanged feature cache')
    x = np.load(source / 'features.npy', allow_pickle=False)
    audit.check(x.dtype == np.float32 and x.shape == (372, 2304), 'Fixed cache dtype and shape')
    with np.load(source / 'predictions.npz', allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    audit.exact(saved['ids'], feature_record['ids'], 'Frozen row order')
    audit.check(len(set(saved['artists'])) == 250, 'Fixed artist count')
    baseline = predict_app.load_bundle()
    correction = predict_app.load_correction(baseline, predict_app.METADATA,
        predict_app.WEIGHTS, predict_app.PLAN, predict_app.CORRECTION)
    calculated = {}
    for name, bundle in [('baseline', baseline), ('candidate', candidate)]:
        z = logits(x, bundle)
        raw = np.exp(-np.logaddexp(0., -z))
        probability = raw.copy()
        if name == 'baseline':
            for j, offset in enumerate(correction['logit_offsets']):
                if offset != 0.:
                    probability[:, j] = np.exp(-np.logaddexp(0., -(z[:, j] + offset)))
        calculated[name] = {'logits': z, 'raw': raw, 'probability': probability}
        for key, values in calculated[name].items():
            audit.exact(values, saved[name + '__' + key], name + ' exact ' + key)
        for policy in ['f1', 'precision_target']:
            decision = raw >= np.asarray(bundle['thresholds'][policy])
            audit.exact(decision, saved[name + '__' + policy + '__decision'], name + ' decision ' + policy)
    for policy in ['f1', 'precision_target']:
        thresholds = candidate['thresholds'][policy]
        for index, feature in enumerate(x):
            encoder = CachedEncoder(feature)
            result = application_predict.classify('cached:' + str(saved['ids'][index]),
                policy=policy, encoder=encoder, compare_baseline=True)
            context = policy + '/' + str(saved['ids'][index])
            audit.check(encoder.calls == 1, context + '/single extraction')
            audit.check(result['application_version'] == '1.2.0'
                and result['score_mode'] == result['status'] == 'validated_candidate', context + '/release')
            audit.check(result['validation'] == release['validation']
                and result['candidate_id'] == release['candidate_id']
                and result['model_weights_sha256'] == release['model_weights_sha256'], context + '/identity')
            audit.check(result['baseline']['application_version'] == '1.1.0'
                and result['baseline']['shared_feature'] is True, context + '/baseline')
            for name, details in [('candidate', result['scores']), ('baseline', result['baseline']['scores'])]:
                audit.check([row['label'] for row in details] == list(predict_app.LABELS), context + '/' + name + '/labels')
                for j, row in enumerate(details):
                    audit.check(row['score'] == round(float(saved[name + '__probability'][index, j]), 4),
                                context + '/' + name + '/score')
                    audit.check(row['raw_score'] == round(float(saved[name + '__raw'][index, j]), 4),
                                context + '/' + name + '/raw score')
                    audit.check(row['selected'] is bool(saved[name + '__' + policy + '__decision'][index, j]),
                                context + '/' + name + '/decision')
                target = result if name == 'candidate' else result['baseline']
                audit.check(target['predicted_tags'] == [row['label'] for row in details if row['selected']],
                            context + '/' + name + '/tags')
            audit.exact([r['threshold'] for r in result['scores']],
                        [round(float(t), 4) for t in thresholds], context + '/candidate thresholds')
    pipeline.verify_frozen(source)
    return {'state': 'passed', 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'application_version': application_release.APP_VERSION, 'candidate_id': release['candidate_id'],
        'checks': audit.checks, 'tracks': len(x), 'artists': len(set(saved['artists'])),
        'policies': ['f1', 'precision_target'], 'public_entrypoint_calls': 2 * len(x),
        'full_precision_predictions_exact': True, 'decisions_and_display_scores_exact': True,
        'one_feature_extraction_per_comparison': True, 'frozen_research_unchanged': True,
        'no_training_or_audio_downloads': True, 'real_audio_extraction': False,
        'application_release_sha256': sha(ROOT / 'artifacts/application_release.json'),
        'application_entrypoint_sha256': sha(ROOT / 'application_predict.py'),
        'release_loader_sha256': sha(ROOT / 'application_release.py'),
        'candidate_weights_sha256': release['model_weights_sha256'],
        'validation_predictions_sha256': sha(source / 'predictions.npz'),
        'verifier_sha256': sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = verify(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
