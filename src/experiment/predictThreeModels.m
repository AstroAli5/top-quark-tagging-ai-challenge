function probabilities = predictThreeModels(models,jets,cfg)
%PREDICTTHREEMODELS Identical jets, model-specific representations.
    n = models.cnn.cfg.imageSize;
    images = zeros(n,n,1,numel(jets),'single');
    features = cell(size(jets)); adjacency = features;
    for j = 1:numel(jets)
        images(:,:,:,j) = single(buildJetImage(jets{j},n));
        [f,a] = buildJetGraph(jets{j},models.sage.cfg.kNeighbors);
        features{j} = single(f); adjacency{j} = sparse(a);
    end
    probabilities = [predictCNN(models.cnn.netCNN,images,models.cnn.classNames,cfg.cnnBatchSize), ...
        predictGraphSAGE(models.sage.parameters,features,adjacency,cfg.graphBatchSize), ...
        predictWinnerReference(models.reference,jets,cfg.winnerBatchSize)];
end
