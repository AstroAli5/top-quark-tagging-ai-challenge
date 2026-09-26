"""Verify a datastore CNN run against its saved scores and official test rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from prepare_parquet import sha256
from summarize_experiment import auc_influences

def verify(run, raw_test):
    run, raw_test = Path(run), Path(raw_test)
    report = json.loads((run/'report.json').read_text())
    m = report['manifest']
    test = m['partitions']['test']
    if not m['verified_official_source'] or sha256(raw_test) != test['source_sha256']:
        raise ValueError('Official test source identity mismatch')
    scores = pd.read_csv(run/'test_predictions.csv')
    saved = loadmat(run/'test_predictions.mat')
    n = report['testJets']
    if len(scores) != n or not np.array_equal(scores.sourceRows, np.arange(n)):
        raise ValueError('Missing or reordered test rows')
    for name in ['sourceRows','labels','probabilities']:
        if not np.allclose(scores[name],saved[name].ravel(),rtol=0,atol=1e-12):
            raise ValueError('CSV and MAT scores differ')
    p = scores.probabilities.to_numpy()
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Invalid probabilities')
    for start in range(0,n,5000):
        stop = min(start+5000,n)
        labels = pd.read_hdf(raw_test,key='table',start=start,stop=stop)['is_signal_new']
        if not np.array_equal(labels.to_numpy(),scores.labels.iloc[start:stop]):
            raise ValueError('Saved labels disagree with official rows')
    auc = auc_influences(p,scores.labels.to_numpy())[0]
    accuracy = float(np.mean((p>=.5)==scores.labels.to_numpy()))
    if not np.allclose([auc,accuracy],[report['auc'],report['accuracy']],rtol=0,atol=1e-10):
        raise ValueError('Metrics disagree with predictions')
    if sha256(run/'cnn_model.mat') != report['checkpointSHA256']:
        raise ValueError('Checkpoint checksum mismatch')
    for part,key in [('train','trainJets'),('val','validationJets'),('test','testJets')]:
        if report[key] != m['partitions'][part]['selected_rows']:
            raise ValueError('Partition counts disagree')
    if report['fullOfficialTraining'] != (report['trainJets']==1211000):
        raise ValueError('Incorrect full-training claim')
    if report['fullOfficialTest'] != (n==404000):
        raise ValueError('Incorrect full-test claim')
    result = dict(verified=True,test_jets=n,accuracy=accuracy,auc=auc,
                  checkpoint_sha256=report['checkpointSHA256'],
                  predictions_sha256=sha256(run/'test_predictions.mat'),
                  source_sha256=test['source_sha256'])
    (run/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--raw-test',type=Path,required=True)
    args=parser.parse_args()
    verify(args.run,args.raw_test)
