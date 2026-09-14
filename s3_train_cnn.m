function s3_train_cnn(cfg)
%S3_TRAIN_CNN Train a compact CNN with normalized classification outputs.
    if nargin < 1, cfg = projectConfig; end
    fprintf('Step 3/7: Training CNN...\n');
    data = load(fullfile(cfg.dataDir,'jet_images.mat'));
    rng(cfg.cnnSeed,'twister');
    XTrain = data.jetImages(:,:,:,data.idxTrain);
    XVal = data.jetImages(:,:,:,data.idxVal);
    % Persist the exact category order used by cross-entropy training.
    YTrain = categorical(data.labels(data.idxTrain),[0 1],{'0','1'});
    YVal = categorical(data.labels(data.idxVal),[0 1],{'0','1'});
    classNames = categories(YTrain);
    layers = [
        imageInputLayer([size(XTrain,1) size(XTrain,2) 1],Normalization='zscore')
        convolution2dLayer(3,16,Padding='same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,Stride=2)
        convolution2dLayer(3,32,Padding='same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,Stride=2)
        convolution2dLayer(3,64,Padding='same')
        batchNormalizationLayer
        reluLayer
        globalAveragePooling2dLayer
        fullyConnectedLayer(2)
        softmaxLayer
    ];
    options = trainingOptions('adam',MaxEpochs=cfg.cnnEpochs, ...
        MiniBatchSize=cfg.cnnBatchSize,InitialLearnRate=1e-3, ...
        Shuffle='every-epoch',ValidationData={XVal,YVal}, ...
        ValidationFrequency=max(1,ceil(numel(YTrain)/cfg.cnnBatchSize)), ...
        OutputNetwork='best-validation-loss', ...
        ExecutionEnvironment=cfg.executionEnvironment,Plots='none',Verbose=false);
    [netCNN,trainingInfo] = trainnet(XTrain,YTrain,layers,'crossentropy',options);
    datasetId = data.datasetId;
    if ~isfolder(cfg.modelsDir), mkdir(cfg.modelsDir); end
    save(fullfile(cfg.modelsDir,'cnn_model.mat'), ...
        'netCNN','classNames','datasetId','cfg','trainingInfo');
    fprintf('Saved CNN model.\n');
end
