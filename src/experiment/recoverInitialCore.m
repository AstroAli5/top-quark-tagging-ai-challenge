function recoverInitialCore(seed)
%RECOVERINITIALCORE Evaluate the completed checkpoints from the timed-out run.
% The workflow restores the ORIGINAL prepared artifact, not a new conversion.
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    officialDir = fullfile(root,'data','official');
    output = fullfile(root,'runs',sprintf('core_%d',seed));
    manifest = jsondecode(fileread(fullfile(officialDir,'manifest.json')));
    assert(strcmp(manifest.fitting_sha256,'b8911d9b5646ff8e031c1baf2946f0c3b532ba382574540d2f37679575f16a6e'), ...
        'Recovery requires the original verified fitting artifact.');
    assert(manifest.train_count == 50000 && manifest.val_count == 10000 && manifest.test_count == 404000);
    a = load(fullfile(output,'models','cnn_model.mat'));
    b = load(fullfile(output,'models','graphsage_model.mat'));
    verifyDatasetIds(a,b);
    assert(isequal(a.cfg,b.cfg) && a.cfg.cnnSeed == seed && a.cfg.graphSeed == seed);
    assert(a.cfg.cnnEpochs == 12 && a.cfg.graphEpochs == 12);
    cfg = a.cfg;
    cfg.dataDir = fullfile(output,'data'); cfg.modelsDir = fullfile(output,'models');
    cfg.resultsDir = fullfile(output,'results'); cfg.inputFile = fullfile(officialDir,'fitting.mat');
    cfg.experimentModels = {'CNN','GraphSAGE'};
    cfg.recoveredFromWorkflow = '35042290356';
    cfg.trainingCommit = '0f7ae539735f1aa5ee113c84bf36a17f4f366e2c';
    % Elapsed start/save timestamps in the original public MATLAB job logs.
    seeds = [101 202 303];
    times = [620.948 407.085;633.243 356.765;440.071 254.135];
    index = find(seeds == seed,1); assert(~isempty(index),'Unexpected recovery seed.');
    evaluateOfficialTest(cfg,officialDir,manifest,seed,times(index,:));
end
