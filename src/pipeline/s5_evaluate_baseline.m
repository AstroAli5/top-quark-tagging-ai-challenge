function s5_evaluate_baseline(cfg)
%S5_EVALUATE_BASELINE Save clean held-out probabilities, metrics, and ROC.
    if nargin < 1, cfg = projectConfig; end
    fprintf('Step 5/7: Evaluating clean test jets...\n');
    images = load(fullfile(cfg.dataDir,'jet_images.mat'));
    graphs = load(fullfile(cfg.dataDir,'jet_graphs.mat'));
    cnn = load(fullfile(cfg.modelsDir,'cnn_model.mat'));
    sage = load(fullfile(cfg.modelsDir,'graphsage_model.mat'));
    verifyDatasetIds(images,graphs,cnn,sage);
    assert(isequal(images.idxTest,graphs.idxTest) && isequal(images.labels,graphs.labels), ...
        'Image and graph test splits must match.');
    labelsTest = images.labels(images.idxTest);
    probCNN = predictCNN(cnn.netCNN,images.jetImages(:,:,:,images.idxTest), ...
        cnn.classNames,cfg.cnnBatchSize);
    probSAGE = predictGraphSAGE(sage.parameters,graphs.jetNodeFeatures(graphs.idxTest), ...
        graphs.jetAdjacency(graphs.idxTest),cfg.graphBatchSize);
    accCNN = mean((probCNN >= 0.5) == labelsTest);
    accSAGE = mean((probSAGE >= 0.5) == labelsTest);
    [tprCNN,fprCNN,aucCNN] = computeROC(probCNN,labelsTest);
    [tprSAGE,fprSAGE,aucSAGE] = computeROC(probSAGE,labelsTest);
    resultsTable = table(["CNN";"GraphSAGE"],[accCNN;accSAGE],[aucCNN;aucSAGE], ...
        VariableNames={'Model','Accuracy','AUC'});
    if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
    writetable(resultsTable,fullfile(cfg.resultsDir,'baseline_comparison.csv'));
    datasetId = images.datasetId;
    idxTest = images.idxTest;
    save(fullfile(cfg.resultsDir,'baseline_predictions.mat'), ...
        'probCNN','probSAGE','labelsTest','idxTest','datasetId');
    disp(resultsTable);
    fig = figure('Visible','off');
    cleanup = onCleanup(@() close(fig));
    plot(fprCNN,tprCNN,'LineWidth',1.5); hold on;
    plot(fprSAGE,tprSAGE,'LineWidth',1.5);
    plot([0 1],[0 1],'k--');
    xlabel('False positive rate'); ylabel('True positive rate');
    legend("CNN (AUC=" + round(aucCNN,3) + ")", ...
        "GraphSAGE (AUC=" + round(aucSAGE,3) + ")",'Random',Location='southeast');
    title('Clean held-out test set'); grid on;
    saveas(fig,fullfile(cfg.resultsDir,'roc_baseline.png'));
end
