function evaluate_winner_comparison(cfg)
%EVALUATE_WINNER_COMPARISON Paired clean/noisy evaluation without retraining.
    if nargin < 1, cfg = projectConfig; end
    split = load(fullfile(cfg.dataDir,'jet_split.mat'));
    cnn = load(fullfile(cfg.modelsDir,'cnn_model.mat'));
    sage = load(fullfile(cfg.modelsDir,'graphsage_model.mat'));
    winner = load(fullfile(cfg.modelsDir,'winner_reference.mat'));
    verifyDatasetIds(split,cnn,sage,winner);
    testJets = split.jetFourVectors(split.idxTest);
    labelsTest = split.labels(split.idxTest);
    noiseLevels = unique([0 cfg.noiseLevels(:).'],'stable');
    names = ["CNN";"GraphSAGE";"ResNeXt-SE reference"];
    accuracy = zeros(numel(noiseLevels),3); auc = accuracy;
    if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
    rocFigure = figure('Visible','off');
    cleanupROC = onCleanup(@() close(rocFigure));
    for level = 1:numel(noiseLevels)
        rng(cfg.noiseSeed,'twister');
        noisyJets = cell(size(testJets));
        images = zeros(split.cfg.imageSize,split.cfg.imageSize,1,numel(testJets),'single');
        features = cell(size(testJets)); adjacency = features;
        for j = 1:numel(testJets)
            noisyJets{j} = injectDetectorNoise(testJets{j},noiseLevels(level));
            images(:,:,:,j) = single(buildJetImage(noisyJets{j},split.cfg.imageSize));
            [features{j},adjacency{j}] = buildJetGraph(noisyJets{j},split.cfg.kNeighbors);
            features{j} = single(features{j}); adjacency{j} = sparse(adjacency{j});
        end
        probabilities = [predictCNN(cnn.netCNN,images,cnn.classNames,cfg.cnnBatchSize), ...
            predictGraphSAGE(sage.parameters,features,adjacency,cfg.graphBatchSize), ...
            predictWinnerReference(winner,noisyJets,cfg.winnerBatchSize)];
        for model = 1:3
            accuracy(level,model) = mean((probabilities(:,model) >= 0.5) == labelsTest);
            [tpr,fpr,auc(level,model)] = computeROC(probabilities(:,model),labelsTest);
            if level == 1
                figure(rocFigure); plot(fpr,tpr,LineWidth=1.5); hold on;
            end
        end
        if level == 1
            cleanProbabilities = probabilities;
        end
        fprintf('Three-model comparison: noise %.2f, AUCs %.4f / %.4f / %.4f\n', ...
            noiseLevels(level),auc(level,1),auc(level,2),auc(level,3));
    end
    comparison = table(names,accuracy(1,:).',auc(1,:).', ...
        VariableNames={'Model','Accuracy','AUC'});
    writetable(comparison,fullfile(cfg.resultsDir,'winner_comparison.csv'));
    noise = table(noiseLevels.',accuracy(:,1),auc(:,1),accuracy(:,2),auc(:,2), ...
        accuracy(:,3),auc(:,3),VariableNames={'NoiseLevel','CNN_Accuracy','CNN_AUC', ...
        'GraphSAGE_Accuracy','GraphSAGE_AUC','Reference_Accuracy','Reference_AUC'});
    writetable(noise,fullfile(cfg.resultsDir,'winner_robustness.csv'));
    datasetId = split.datasetId; idxTest = split.idxTest;
    save(fullfile(cfg.resultsDir,'winner_predictions.mat'), ...
        'cleanProbabilities','names','labelsTest','idxTest','datasetId');
    figure(rocFigure); plot([0 1],[0 1],'k--');
    xlabel('False positive rate'); ylabel('True positive rate'); grid on;
    legend([names;"Random"],Location='southeast'); title('Shared clean holdout');
    saveas(rocFigure,fullfile(cfg.resultsDir,'winner_roc.png'));
    noiseFigure = figure('Visible','off');
    cleanupNoise = onCleanup(@() close(noiseFigure));
    tiledlayout(1,2);
    nexttile; plot(noiseLevels,accuracy,'-o',LineWidth=1.5);
    xlabel('Synthetic component smearing (fractional)'); ylabel('Accuracy'); grid on;
    legend(names,Location='southwest');
    nexttile; plot(noiseLevels,auc,'-o',LineWidth=1.5);
    xlabel('Synthetic component smearing (fractional)'); ylabel('AUC'); grid on;
    legend(names,Location='southwest');
    saveas(noiseFigure,fullfile(cfg.resultsDir,'winner_robustness.png'));
    metadata = struct('evaluatedAtUTC',char(datetime('now','TimeZone','UTC')), ...
        'matlabVersion',version,'datasetId',datasetId,'sourceInfo',split.sourceInfo, ...
        'trainJets',numel(split.idxTrain),'validationJets',numel(split.idxVal), ...
        'testJets',numel(idxTest),'noiseLevels',noiseLevels,'noiseSeed',cfg.noiseSeed, ...
        'cnnConfiguration',cnn.cfg,'graphConfiguration',sage.cfg, ...
        'referenceConfiguration',winner.cfg,'normalization',winner.normalization, ...
        'protocol','Internal holdout from input subset; not official test partition', ...
        'reference','https://github.com/adit-smoak/Top-Quark-Tagging-Using-Deep-CNN');
    fid = fopen(fullfile(cfg.resultsDir,'winner_metadata.json'),'w');
    if fid < 0, error('topquark:WriteFailed','Cannot write winner metadata.'); end
    cleanupFile = onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(metadata,PrettyPrint=true));
    disp(comparison);
end
