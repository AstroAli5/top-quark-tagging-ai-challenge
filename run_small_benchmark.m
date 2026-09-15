function run_small_benchmark
%RUN_SMALL_BENCHMARK Bounded real-data feasibility run, not a final benchmark.
% First convert 2,000 rows of official train.h5 to data/jets_real_2000.mat.
    cfg = projectConfig;
    cfg.inputFile = fullfile(cfg.dataDir,'jets_real_2000.mat');
    cfg.cnnEpochs = 3;
    cfg.graphEpochs = 3;
    cfg.winnerEpochs = 3;
    cfg.cnnBatchSize = 64;
    cfg.graphBatchSize = 32;
    cfg.winnerBatchSize = 32;
    cfg.noiseLevels = [0 0.1 0.2];
    cfg.radiusFractions = [1 0.5];
    run_all(cfg);
    run_winner_comparison(cfg);
end
