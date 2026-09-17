function cfg = projectConfig(outputRoot)
%PROJECTCONFIG Portable defaults. Edit a returned struct to change a run.
%   cfg = projectConfig; run_all(cfg)
%   cfg = projectConfig(tempname) isolates all generated data and outputs.
    setupProject;
    if nargin < 1
        outputRoot = fileparts(mfilename('fullpath'));
    end
    cfg.dataDir = fullfile(outputRoot,'data');
    cfg.modelsDir = fullfile(outputRoot,'models');
    cfg.resultsDir = fullfile(outputRoot,'results');
    cfg.inputFile = fullfile(cfg.dataDir,'jets_real_50k.mat');
    cfg.imageSize = 32;
    cfg.kNeighbors = 6;
    cfg.splitSeed = 42;
    cfg.cnnSeed = 101;
    cfg.graphSeed = 102;
    cfg.noiseSeed = 7;
    cfg.permutationSeed = 11;
    cfg.cnnEpochs = 15;
    cfg.graphEpochs = 3;
    cfg.cnnBatchSize = 128;
    cfg.graphBatchSize = 64;
    cfg.hiddenSize = 32;
    cfg.graphLearnRate = 0.01;
    cfg.noiseLevels = [0 0.02 0.05 0.10 0.20 0.35];
    cfg.radiusFractions = [1 0.75 0.5 0.35 0.2];
    % CPU avoids mixed sparse-CPU / GPU operations and needs no GPU toolbox.
    cfg.executionEnvironment = 'cpu';
    % Independent, compact reference inspired by the 2025 winning project.
    cfg.winnerImageSize = 37;
    cfg.winnerExtent = 1.6;
    cfg.winnerMaxParticles = 35;
    cfg.winnerWidths = [32 64 128];
    cfg.winnerGroups = 4;
    cfg.winnerSeed = 103;
    cfg.winnerEpochs = 12;
    cfg.winnerBatchSize = 64;
    cfg.winnerLearnRate = 5e-3;
    cfg.winnerCacheImages = false;
end
