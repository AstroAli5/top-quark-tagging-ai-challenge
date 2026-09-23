"""Verify compatibility of the explicitly specified five-seed core extension."""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess
import re

BASELINE = '0ffc6e86e8b7faf08a73edb5f9ee9628e68fce08'
ORIGINAL_TRAINING = '0f7ae539735f1aa5ee113c84bf36a17f4f366e2c'
TRAIN_PATHS = ['src/core', 'src/pipeline', 'projectConfig.m', 'setupProject.m',
               'scripts/prepare_official.py', 'scripts/convert_dataset.py', 'scripts/download_dataset.py']
EVALUATION_PATHS = ['src/core', 'src/experiment/readJetChunk.m',
                    'src/experiment/predictThreeModels.m', 'src/experiment/evaluateOfficialTest.m']


def git(*args):
    return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1], text=True)


def fingerprint(commit, paths):
    # Compare git blob identities, not untrusted declarations in run metadata.
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Expected an immutable Git commit SHA')
    listing = git('ls-tree', '-r', commit, '--', *paths)
    if not listing: raise ValueError(f'Missing source tree for {commit}')
    return hashlib.sha256(listing.encode()).hexdigest()


def semantic_manifest(manifest):
    return {**{k: v for k, v in manifest.items() if k not in ['fitting_sha256', 'test_chunks']},
            'test_chunks': [{k: v for k, v in chunk.items() if k != 'sha256'}
                            for chunk in manifest['test_chunks']]}


def configuration(config):
    ignored = {'dataDir', 'modelsDir', 'resultsDir', 'inputFile', 'cnnSeed', 'graphSeed',
               'winnerSeed', 'recoveredFromWorkflow', 'trainingCommit'}
    return {k: v for k, v in config.items() if k not in ignored}


def verify_compatibility(metadata):
    if sorted(m['trainingSeed'] for m in metadata) != [101, 202, 303, 404, 505]:
        raise ValueError('The core extension requires exactly seeds 101, 202, 303, 404, 505')
    baseline_train = fingerprint(BASELINE, TRAIN_PATHS)
    baseline_eval = fingerprint(BASELINE, EVALUATION_PATHS)
    allowed_controllers = {git('rev-parse', f'{c}:run_experiment.m').strip()
                           for c in [ORIGINAL_TRAINING, BASELINE]}
    first = metadata[0]
    records = []
    for m in metadata:
        manifest = m['manifest']; cfg = m['configuration']
        if m['models'] != ['CNN', 'GraphSAGE']:
            raise ValueError('The five-seed report is for core models only')
        if cfg['cnnSeed'] != m['trainingSeed'] or cfg['graphSeed'] != m['trainingSeed']:
            raise ValueError('Configured training seeds disagree with the run label')
        if (manifest['train_count'], manifest['val_count'], manifest['test_count']) != (50000, 10000, 404000):
            raise ValueError('The fitting or test sample changed')
        if semantic_manifest(manifest) != semantic_manifest(first['manifest']):
            raise ValueError('Source data or partition selections changed')
        if configuration(cfg) != configuration(first['configuration']):
            raise ValueError('Training or evaluation settings changed')
        if m['matlabVersion'] != first['matlabVersion']:
            raise ValueError('MATLAB versions differ; compatibility needs separate review')
        train_commit = m.get('trainingProvenance', {}).get('codeCommit', m['codeCommit'])
        train_hash = fingerprint(train_commit, TRAIN_PATHS)
        evaluation_hash = fingerprint(m['codeCommit'], EVALUATION_PATHS)
        controller = git('rev-parse', f'{train_commit}:run_experiment.m').strip()
        if train_hash != baseline_train or evaluation_hash != baseline_eval or controller not in allowed_controllers:
            raise ValueError('Source implementation differs from the reviewed baseline')
        records.append({'seed': m['trainingSeed'], 'training_commit': train_commit,
                        'evaluation_commit': m['codeCommit'], 'workflow_run': m['workflowRun'],
                        'training_source_sha256': train_hash, 'evaluation_source_sha256': evaluation_hash,
                        'controller_git_blob': controller, 'prepared_fitting_sha256': manifest['fitting_sha256']})
    return {'status': 'compatible', 'reviewed_baseline': BASELINE, 'runs': records,
            'note': 'Prepared MAT header hashes may differ. Verified original source hashes, source row selections, settings, MATLAB version and training/evaluation source identities. Original and core-only run controllers were reviewed separately.'}
