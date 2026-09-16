function run_experiment(seed,officialDir,outputRoot,overrides)
%RUN_EXPERIMENT Fit on official train/val, then evaluate official test chunks.
%   run_experiment(101) uses data prepared by scripts/prepare_official.py.
    setupProject;
    root = fileparts(mfilename('fullpath'));
    if nargin < 1, seed = 101; end
    if nargin < 2, officialDir = fullfile(root,'data','official'); end
    if nargin < 3, outputRoot = fullfile(root,'runs',sprintf('seed_%d',seed)); end
    if nargin < 4, overrides = struct; end
    cfg = projectConfig(outputRoot);
    cfg.inputFile = fullfile(officialDir,'fitting.mat');
    cfg.cnnSeed = seed; cfg.graphSeed = seed; cfg.winnerSeed = seed;
    cfg.cnnEpochs = 12; cfg.graphEpochs = 12; cfg.winnerEpochs = 12;
    cfg.winnerCacheImages = true;
    cfg.cnnBatchSize = 64; cfg.graphBatchSize = 32; cfg.winnerBatchSize = 64;
    cfg.noiseLevels = [0 0.05 0.10 0.20 0.35];
    cfg.noiseSeeds = [7 17 27]; cfg.noiseTestJets = 10000;
    for name = reshape(fieldnames(overrides),1,[])
        cfg.(name{1}) = overrides.(name{1});
    end
    if isfolder(cfg.modelsDir) && ~isempty(dir(fullfile(cfg.modelsDir,'*.mat')))
        error('topquark:ExistingRun','Choose a new output folder; existing trained models are preserved.');
    end
    manifest = jsondecode(fileread(fullfile(officialDir,'manifest.json')));
    if ~strcmp(manifest.dataset,'10.5281/zenodo.2603256')
        error('topquark:InvalidPartition','Unexpected dataset manifest.');
    end
    started = tic;
    s1_prepare_data(cfg); s2_build_representations(cfg);
    split = load(fullfile(cfg.dataDir,'jet_split.mat'),'idxTrain','idxVal','idxTest');
    if numel(split.idxTrain) ~= manifest.train_count || numel(split.idxVal) ~= manifest.val_count || ~isempty(split.idxTest)
        error('topquark:InvalidPartition','Official fitting counts changed; inspect excluded jets before reporting.');
    end
    clear split;
    times = zeros(1,3);
    timer = tic; s3_train_cnn(cfg); times(1) = toc(timer);
    timer = tic; s4_train_graphsage(cfg); times(2) = toc(timer);
    timer = tic; train_winner_reference(cfg); times(3) = toc(timer);
    evaluateOfficialTest(cfg,officialDir,manifest,seed,times);
    fprintf('Official experiment seed %d completed in %.1f seconds.\n',seed,toc(started));
end
