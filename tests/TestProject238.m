function tests = TestProject238
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.root = root;
    testCase.applyFixture(matlab.unittest.fixtures.PathFixture(root));
    for folder = {'core','bigdata','pipeline','reference','experiment'}
        testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,'src',folder{1})));
    end
    pythonRoot = getenv('pythonLocation');
    if ~isempty(pythonRoot)
        pyenv(Version=fullfile(pythonRoot,'bin','python'),ExecutionMode="OutOfProcess");
    else
        pyenv(ExecutionMode="OutOfProcess");
    end
end

function testTallImagesAndDatastoreTraining(testCase)
    folder = tempname; mkdir(folder); cleanup = onCleanup(@() rmdir(folder,'s'));
    cfg = project238Config(folder);
    cfg.epochs = 1; cfg.batchSize = 5; cfg.chunkRows = 4;
    cfg.outputDir = fullfile(folder,'fit');
    scripts = fullfile(testCase.TestData.root,'scripts');
    pyrun(["import sys, pathlib, numpy as np, pandas as pd", ...
        "sys.path.insert(0,str(scripts))", ...
        "from prepare_parquet import prepare", ...
        "from convert_dataset import PARTICLE_COLUMNS", ...
        "raw = pathlib.Path(str(raw)); raw.mkdir(parents=True,exist_ok=True)", ...
        "x = np.zeros((12,800),dtype=np.float32)", ...
        "x[:,:12] = [10,8,3,2,15,9,-4,6,8,5,2,-3]", ...
        "x[0,4:] = 0", ...
        "df = pd.DataFrame(x,columns=PARTICLE_COLUMNS)", ...
        "df['is_signal_new'] = np.arange(12)%2", ...
        "for part in ['train','val','test']: df.to_hdf(raw/(part+'.h5'),key='table',format='table')", ...
        "prepare(raw,str(output),{'train':12,'val':12,'test':12},4,download_sources=False)"], ...
        scripts=string(scripts),raw=string(cfg.rawDir),output=string(fullfile(cfg.dataDir,'parquet')));
    manifest = parquetJetsToImages(cfg);
    [ds,rows] = jetImageDatastore(fullfile(cfg.dataDir,'images','test'),manifest.partitions.test,5);
    verifyEqual(testCase,rows,(0:11).');
    verifyEqual(testCase,double(ds.Labels=="signal"),mod(rows,2));
    first = imread(ds.Files{1});
    verifyClass(testCase,first,'single');
    verifyEqual(testCase,first,single(buildJetImage([10 8 3 2],32)));
    verifyEqual(testCase,imread(ds.Files{2}), ...
        single(buildJetImage([10 8 3 2;15 9 -4 6;8 5 2 -3],32)));
    verifyEqual(testCase,parquetJetsToImages(cfg),manifest);
    before = rng; report = trainProject238(cfg,manifest); verifyEqual(testCase,rng,before);
    verifyEqual(testCase,[report.trainJets report.validationJets report.testJets],[12 12 12]);
    verifyFalse(testCase,report.fullOfficialTraining); verifyFalse(testCase,report.fullOfficialTest);
    verifyTrue(testCase,isfinite(report.auc));
    predictions = load(fullfile(cfg.outputDir,'test_predictions.mat'));
    verifyEqual(testCase,predictions.sourceRows,rows);
    verifyEqual(testCase,predictions.labels,mod(rows,2));
    verifyError(testCase,@() trainProject238(cfg,manifest),'topquark:ExistingRun');
    delete(ds.Files{end});
    verifyError(testCase,@() jetImageDatastore(fullfile(cfg.dataDir,'images','test'), ...
        manifest.partitions.test,5),'topquark:ImageCoverage');
end

function testShippedCheckpointPredictionsAndMatlabSummary(testCase)
    result = verify_results("all");
    verifySize(testCase,result,[3 3]);
    folder = tempname; mkdir(folder); cleanup = onCleanup(@() rmdir(folder,'s'));
    input = fullfile(testCase.TestData.root,'experiments','official-study');
    review = run_submission(folder);
    summary = review.study;
    cnn = summary(summary.Model=="CNN",:);
    verifyEqual(testCase,cnn.AccuracyMean,0.9105866336633662,'AbsTol',1e-12);
    verifyEqual(testCase,cnn.AUCMean,0.9692862903638857,'AbsTol',1e-12);
    oracle = readtable(fullfile(input,'paired_auc.csv'),TextType='string');
    verifyEqual(testCase,review.paired.Difference,oracle.Difference);
    verifyEqual(testCase,review.paired.MeanAUCDifference,oracle.MeanAUC_Difference,'AbsTol',1e-12);
    verifyEqual(testCase,review.paired.AUCSeedCI_Low,oracle.TrainingSeedCI_Low,'AbsTol',1e-12);
    verifyEqual(testCase,review.paired.AUCSeedCI_High,oracle.TrainingSeedCI_High,'AbsTol',1e-12);
    verifyTrue(testCase,isfile(fullfile(folder,'matlab_study_summary.png')));
end

function testSummaryRejectsUnmatchedSeeds(testCase)
    folder = tempname; mkdir(folder); cleanup = onCleanup(@() rmdir(folder,'s'));
    input = fullfile(testCase.TestData.root,'experiments','official-study');
    clean = readtable(fullfile(input,'per_seed_clean.csv'),TextType='string');
    clean(clean.Model=="GraphSAGE" & clean.Seed==101,:) = [];
    writetable(clean,fullfile(folder,'per_seed_clean.csv'));
    copyfile(fullfile(input,'per_seed_noise.csv'),fullfile(folder,'per_seed_noise.csv'));
    verifyError(testCase,@() summarize_matlab(folder,fullfile(folder,'output')), ...
        'topquark:UnpairedSeeds');
end
