function tests = TestPipeline
% Synthetic integration test. Its metrics are not physics benchmark results.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    testCase.applyFixture(matlab.unittest.fixtures.PathFixture(root));
end

function testAllSevenStagesFromAnotherDirectory(testCase)
    folder = tempname;
    mkdir(folder);
    original = pwd;
    cleanup = onCleanup(@() restoreAndRemove(original,folder));
    cfg = projectConfig(folder);
    mkdir(cfg.dataDir);
    cfg.cnnEpochs = 1;
    cfg.graphEpochs = 1;
    cfg.cnnBatchSize = 8;
    cfg.graphBatchSize = 7; % exercises a partial batch
    cfg.hiddenSize = 8;
    cfg.noiseLevels = [0 0.1];
    cfg.radiusFractions = [1 0.5];
    rng(2026);
    labels = repmat([0;1],20,1);
    particleData = zeros(40,800,'single');
    for j = 1:40
        n = 6+mod(j,5);
        pt = 5+30*rand(n,1);
        eta = (0.05+0.2*labels(j))*randn(n,1);
        phi = 0.15*randn(n,1);
        fv = [pt.*cosh(eta),pt.*cos(phi),pt.*sin(phi),pt.*sinh(eta)];
        particleData(j,1:4*n) = reshape(fv.',1,[]);
    end
    provenance_json = jsonencode(struct('source','synthetic integration fixture'));
    save(cfg.inputFile,'particleData','labels','provenance_json');
    cd(folder);
    run_all(cfg);
    verifyEqual(testCase,pwd,folder);
    baseline = readtable(fullfile(cfg.resultsDir,'baseline_comparison.csv'));
    robustness = readtable(fullfile(cfg.resultsDir,'robustness_results.csv'));
    verifyEqual(testCase,height(baseline),2);
    verifyEqual(testCase,robustness{1,2:5}, ...
        [baseline.Accuracy(1) baseline.AUC(1) baseline.Accuracy(2) baseline.AUC(2)], ...
        'AbsTol',1e-10);
    numericResults = robustness{:,2:5};
    verifyTrue(testCase,all(isfinite(numericResults(:))));
    verifyTrue(testCase,all(numericResults(:) >= 0 & numericResults(:) <= 1));
    predictions = load(fullfile(cfg.resultsDir,'baseline_predictions.mat'));
    verifyEqual(testCase,size(predictions.probCNN),size(predictions.labelsTest));
    radius = readtable(fullfile(cfg.resultsDir,'cnn_radial_occlusion.csv'));
    verifyEqual(testCase,radius.AUC(1),baseline.AUC(1),'AbsTol',1e-10);
    importance = readtable(fullfile(cfg.resultsDir,'graphsage_feature_importance.csv'));
    verifyEqual(testCase,height(importance),4);
    metadata = jsondecode(fileread(fullfile(cfg.resultsDir,'run_metadata.json')));
    verifyEqual(testCase,metadata.trainJets+metadata.validationJets+metadata.testJets,40);
    for name = {'roc_baseline.png','robustness_curves.png', ...
            'graphsage_feature_importance.png','cnn_radial_occlusion.png'}
        info = dir(fullfile(cfg.resultsDir,name{1}));
        verifyNotEmpty(testCase,info);
        verifyGreaterThan(testCase,info.bytes,0);
    end
end

function restoreAndRemove(original,folder)
    cd(original);
    if isfolder(folder), rmdir(folder,'s'); end
end
