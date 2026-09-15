function run_winner_comparison(cfg)
%RUN_WINNER_COMPARISON Train reference, compare all three on the shared holdout.
% Run run_all(cfg) first to prepare jets and train the original two models.
    if nargin < 1, cfg = projectConfig; end
    split = load(fullfile(cfg.dataDir,'jet_split.mat'));
    cnn = load(fullfile(cfg.modelsDir,'cnn_model.mat'));
    sage = load(fullfile(cfg.modelsDir,'graphsage_model.mat'));
    verifyDatasetIds(split,cnn,sage);
    train_winner_reference(cfg);
    evaluate_winner_comparison(cfg);
end
