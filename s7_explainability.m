function s7_explainability(cfg)
%S7_EXPLAINABILITY Fixed-topology feature permutation and image occlusion.
%   Perturbation sensitivity is not a causal explanation of the classifier.
    if nargin < 1, cfg = projectConfig; end
    fprintf('Step 7/7: Feature permutation and radial occlusion...\n');
    graphs = load(fullfile(cfg.dataDir,'jet_graphs.mat'));
    images = load(fullfile(cfg.dataDir,'jet_images.mat'));
    cnn = load(fullfile(cfg.modelsDir,'cnn_model.mat'));
    sage = load(fullfile(cfg.modelsDir,'graphsage_model.mat'));
    verifyDatasetIds(graphs,images,cnn,sage);
    if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
    featureNames = ["deltaEta";"deltaPhi";"log(pT)";"log(E)"];
    features = graphs.jetNodeFeatures(graphs.idxTest);
    adjacency = graphs.jetAdjacency(graphs.idxTest);
    labelsTest = graphs.labels(graphs.idxTest);
    probBase = predictGraphSAGE(sage.parameters,features,adjacency,cfg.graphBatchSize);
    [~,~,aucBase] = computeROC(probBase,labelsTest);
    % Stack only four feature columns, never a test-set-wide adjacency.
    nodeCounts = cellfun(@(x) size(x,1),features(:));
    allFeatures = vertcat(features{:});
    rng(cfg.permutationSeed,'twister');
    aucDrop = zeros(4,1);
    for f = 1:4
        shuffled = allFeatures;
        shuffled(:,f) = shuffled(randperm(size(shuffled,1)),f);
        perturbed = mat2cell(shuffled,nodeCounts,size(shuffled,2));
        p = predictGraphSAGE(sage.parameters,perturbed,adjacency,cfg.graphBatchSize);
        [~,~,aucPerturbed] = computeROC(p,labelsTest);
        aucDrop(f) = aucBase-aucPerturbed;
    end
    importanceTable = table(featureNames,aucDrop,VariableNames={'Feature','AUCDrop'});
    writetable(importanceTable,fullfile(cfg.resultsDir,'graphsage_feature_importance.csv'));
    fig1 = figure('Visible','off');
    cleanup1 = onCleanup(@() close(fig1));
    bar(aucDrop); xticks(1:4); xticklabels(featureNames);
    ylabel('AUC drop when shuffled'); title('GraphSAGE feature permutation (fixed edges)');
    grid on; saveas(fig1,fullfile(cfg.resultsDir,'graphsage_feature_importance.png'));

    XTest = images.jetImages(:,:,:,images.idxTest);
    labelsImage = images.labels(images.idxTest);
    imageSize = size(XTest,1);
    [x,y] = meshgrid(1:imageSize,1:imageSize);
    center = (imageSize+1)/2;
    radiusMap = hypot(x-center,y-center);
    radiusFractions = cfg.radiusFractions(:);
    aucByRadius = zeros(size(radiusFractions));
    for i = 1:numel(radiusFractions)
        mask = radiusMap <= radiusFractions(i)*max(radiusMap(:));
        masked = XTest .* single(mask);
        p = predictCNN(cnn.netCNN,masked,cnn.classNames,cfg.cnnBatchSize);
        [~,~,aucByRadius(i)] = computeROC(p,labelsImage);
    end
    radiusTable = table(radiusFractions,aucByRadius, ...
        VariableNames={'RadiusFractionKept','AUC'});
    writetable(radiusTable,fullfile(cfg.resultsDir,'cnn_radial_occlusion.csv'));
    fig2 = figure('Visible','off');
    cleanup2 = onCleanup(@() close(fig2));
    plot(radiusFractions,aucByRadius,'-o',LineWidth=1.5);
    xlabel('Fraction of center-to-corner image radius kept'); ylabel('AUC');
    title('CNN radial occlusion'); grid on; set(gca,'XDir','reverse');
    saveas(fig2,fullfile(cfg.resultsDir,'cnn_radial_occlusion.png'));
    fprintf('Saved explainability CSVs and plots.\n');
end
