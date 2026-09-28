"""Small, dependency-light helpers; never import historical training runners."""
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LABELS = ['electronic', 'pop', 'ambient', 'rock']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def ontology():
    plan = read(ROOT / 'experiments/final_holdout_plan.json')
    # Read literals without importing download/audio dependencies.
    literals = {}
    for node in ast.parse((ROOT / 'expanded_model.py').read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('LABELS', 'ONTOLOGY'):
                    literals[target.id] = ast.literal_eval(node.value)
    if plan['labels'] != LABELS or literals != {'LABELS': LABELS, 'ONTOLOGY': plan['ontology']}:
        raise ValueError('Frozen ontology differs from original broad_targets definition')
    return plan['ontology']


def targets(rows):
    import numpy as np
    mapping = ontology()
    return np.array([[int(bool(set(row['tags']) & set(mapping[label])))
                      for label in LABELS] for row in rows], dtype=int)


def development():
    import numpy as np
    plan = read(ROOT / 'experiments/final_holdout_plan.json')
    manifest = read(ROOT / 'data/expanded_manifest.json')['tracks']
    by_id = {r['track_id']: r for r in manifest}
    if len(by_id) != len(manifest):
        raise ValueError('Duplicate manifest IDs')
    rows = [by_id[i] for i in plan['fit_ids']]
    if len(rows) != 1206 or any(r['phase_role'] not in ('train', 'development_validation') for r in rows):
        raise ValueError('Invalid development pool')
    return rows, targets(rows), np.array([r['artist_id'] for r in rows])
