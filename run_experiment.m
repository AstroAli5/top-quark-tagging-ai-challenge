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
    cfg.experimentModels = {'CNN','GraphSAGE','ResNeXt-SE reference'};
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
    names = string(cfg.experimentModels);
    available = ["CNN","GraphSAGE","ResNeXt-SE reference"];
    assert(~isempty(names) && isequal(names,available(ismember(available,names))), ...
        'Select unique models in CNN, GraphSAGE, reference order.');
    trainers = {@s3_train_cnn,@s4_train_graphsage,@train_winner_reference};
    times = zeros(1,numel(names));
    for j = 1:numel(names)
        timer = tic; trainers{find(available == names(j),1)}(cfg); times(j) = toc(timer);
        save(fullfile(cfg.modelsDir,'training_progress.mat'),'names','times');
    end
    evaluateOfficialTest(cfg,officialDir,manifest,seed,times);
    fprintf('Official experiment seed %d completed in %.1f seconds.\n',seed,toc(started));
end
