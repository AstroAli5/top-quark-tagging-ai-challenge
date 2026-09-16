function probabilities = predictThreeModels(models,jets,cfg)
%PREDICTTHREEMODELS Identical jets, model-specific representations.
    probabilities = zeros(numel(jets),0);
    if isfield(models,'cnn')
        n = models.cnn.cfg.imageSize;
        images = zeros(n,n,1,numel(jets),'single');
        for j = 1:numel(jets)
            images(:,:,:,j) = single(buildJetImage(jets{j},n));
        end
        probabilities(:,end+1) = predictCNN(models.cnn.netCNN,images,models.cnn.classNames,cfg.cnnBatchSize);
    end
    if isfield(models,'sage')
        features = cell(size(jets)); adjacency = features;
        for j = 1:numel(jets)
            [f,a] = buildJetGraph(jets{j},models.sage.cfg.kNeighbors);
            features{j} = single(f); adjacency{j} = sparse(a);
        end
        probabilities(:,end+1) = predictGraphSAGE(models.sage.parameters,features,adjacency,cfg.graphBatchSize);
    end
    if isfield(models,'reference')
        probabilities(:,end+1) = predictWinnerReference(models.reference,jets,cfg.winnerBatchSize);
    end
end
