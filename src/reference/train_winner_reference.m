function train_winner_reference(cfg)
%TRAIN_WINNER_REFERENCE Fit the third model using the existing shared split.
    if nargin < 1, cfg = projectConfig; end
    split = load(fullfile(cfg.dataDir,'jet_split.mat'));
    fprintf('Training the 2025-winner-inspired reference on %d jets...\n',numel(split.idxTrain));
    rng(cfg.winnerSeed,'twister');
    normalization = fitWinnerNormalization(split.jetFourVectors(split.idxTrain),cfg);
    % Cache once for larger CPU runs; the default streaming route uses less RAM.
    if isfield(cfg,'winnerCacheImages') && cfg.winnerCacheImages
        makeData = @cachedWinnerDatastore;
    else
        makeData = @winnerDatastore;
    end
    trainData = makeData(split.jetFourVectors(split.idxTrain), ...
        split.labels(split.idxTrain),normalization,cfg);
    valData = makeData(split.jetFourVectors(split.idxVal), ...
        split.labels(split.idxVal),normalization,cfg);
    net = buildWinnerNetwork(cfg);
    options = trainingOptions('adam',MaxEpochs=cfg.winnerEpochs, ...
        MiniBatchSize=cfg.winnerBatchSize,InitialLearnRate=cfg.winnerLearnRate, ...
        LearnRateSchedule='piecewise',LearnRateDropFactor=0.3,LearnRateDropPeriod=4, ...
        L2Regularization=1e-4,Shuffle='every-epoch',ValidationData=valData, ...
        ValidationFrequency=max(1,ceil(numel(split.idxTrain)/cfg.winnerBatchSize)), ...
        OutputNetwork='best-validation',ExecutionEnvironment=cfg.executionEnvironment, ...
        Plots='none',Verbose=false);
    [netWinner,trainingInfo] = trainnet(trainData,net,'crossentropy',options);
    classNames = {'0';'1'};
    datasetId = split.datasetId;
    if ~isfolder(cfg.modelsDir), mkdir(cfg.modelsDir); end
    save(fullfile(cfg.modelsDir,'winner_reference.mat'), ...
        'netWinner','normalization','classNames','datasetId','cfg','trainingInfo');
end
