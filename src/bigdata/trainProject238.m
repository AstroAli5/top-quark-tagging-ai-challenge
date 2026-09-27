function report = trainProject238(cfg,manifest)
%TRAINPROJECT238 CNN training and official-partition testing from disk.
    if isfolder(cfg.outputDir)
        error('topquark:ExistingRun','Choose a new outputDir.');
    end
    mkdir(cfg.outputDir);
    savedRNG = rng; cleanupRNG = onCleanup(@() rng(savedRNG));
    rng(cfg.seed,'twister');
    root = fullfile(cfg.dataDir,'images');
    train = jetImageDatastore(fullfile(root,'train'),manifest.partitions.train,cfg.batchSize);
    val = jetImageDatastore(fullfile(root,'val'),manifest.partitions.val,cfg.batchSize);
    [test,sourceRows] = jetImageDatastore(fullfile(root,'test'),manifest.partitions.test,cfg.batchSize);
    classNames = categories(train.Labels);
    assert(isequal(string(classNames),["background";"signal"]),'Unexpected class order.');
    layers = [
        imageInputLayer([cfg.imageSize cfg.imageSize 1],Normalization='zscore')
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
        softmaxLayer];
    checkpointDir = fullfile(cfg.outputDir,'epoch-checkpoints'); mkdir(checkpointDir);
    options = trainingOptions('adam',MaxEpochs=cfg.epochs,MiniBatchSize=cfg.batchSize, ...
        InitialLearnRate=1e-3,Shuffle='every-epoch',ValidationData=val, ...
        ValidationFrequency=max(1,ceil(numel(train.Files)/cfg.batchSize)), ...
        OutputNetwork='best-validation',ExecutionEnvironment=cfg.executionEnvironment, ...
        CheckpointPath=checkpointDir,Plots='none',Verbose=true);
    timer = tic;
    [netCNN,trainingInfo] = trainnet(train,layers,'crossentropy',options);
    trainingSeconds = toc(timer);
    save(fullfile(cfg.outputDir,'cnn_model.mat'),'netCNN','classNames','trainingInfo','cfg','manifest');
    timer = tic;
    scores = minibatchpredict(netCNN,test,MiniBatchSize=cfg.batchSize, ...
        ExecutionEnvironment=cfg.executionEnvironment,OutputDataFormats='BC');
    if isa(scores,'dlarray'), scores = extractdata(scores); end
    scores = double(gather(scores));
    if ~isequal(size(scores),[numel(test.Files),2]) || any(~isfinite(scores),'all') || ...
            any(scores<0 | scores>1,'all') || any(abs(sum(scores,2)-1)>1e-5)
        error('topquark:InvalidProbabilities','Invalid class probabilities.');
    end
    labels = double(test.Labels=="signal");
    probabilities = scores(:,string(classNames)=="signal");
    [tpr,fpr,aucValue] = computeROC(probabilities,labels);
    accuracy = mean((probabilities>=0.5)==labels);
    evaluationSeconds = toc(timer);
    save(fullfile(cfg.outputDir,'test_predictions.mat'),'probabilities','labels','sourceRows');
    predictionTable = table(sourceRows,labels,probabilities);
    writetable(predictionTable,fullfile(cfg.outputDir,'test_predictions.csv'));
    [~,commit] = system('git rev-parse HEAD');
    report = struct('configuration',cfg,'manifest',manifest,'matlabVersion',version, ...
        'codeCommit',strtrim(commit),'workflowRun',getenv('GITHUB_RUN_ID'), ...
        'trainJets',numel(train.Files),'validationJets',numel(val.Files),'testJets',numel(test.Files), ...
        'accuracy',accuracy,'auc',aucValue,'trainingSeconds',trainingSeconds, ...
        'evaluationSeconds',evaluationSeconds, ...
        'fullOfficialTraining',manifest.verified_official_source && numel(train.Files)==1211000, ...
        'fullOfficialTest',manifest.verified_official_source && numel(test.Files)==404000, ...
        'normalization','imageInputLayer zscore fitted from training datastore only', ...
        'shuffle','all training image filenames shuffled every epoch by trainnet', ...
        'checkpointSHA256',projectFileSHA256(fullfile(cfg.outputDir,'cnn_model.mat')));
    writeProjectJSON(fullfile(cfg.outputDir,'report.json'),report);
    fig = figure('Visible','off'); cleanup = onCleanup(@() close(fig));
    tiledlayout(1,2);
    nexttile; plot(fpr,tpr,LineWidth=1.5); xlabel('False positive rate'); ylabel('True positive rate');
    title(sprintf('CNN ROC: AUC %.5f',aucValue)); grid on;
    nexttile; counts = [sum(labels==0 & probabilities<.5),sum(labels==0 & probabilities>=.5); ...
        sum(labels==1 & probabilities<.5),sum(labels==1 & probabilities>=.5)];
    imagesc(counts); colorbar; xticks(1:2); yticks(1:2);
    xticklabels({'background','signal'}); yticklabels({'background','signal'});
    xlabel('Predicted'); ylabel('True'); title(sprintf('Accuracy %.2f%%',100*accuracy));
    sgtitle(sprintf('MATLAB datastore CNN: %d training / %d test jets, seed %d', ...
        report.trainJets,report.testJets,cfg.seed));
    exportgraphics(fig,fullfile(cfg.outputDir,'matlab_evaluation.png'),Resolution=180);
    disp(report);
end
