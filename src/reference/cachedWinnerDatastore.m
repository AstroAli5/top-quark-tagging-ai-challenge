function ds = cachedWinnerDatastore(jets,labels,normalization,cfg)
%CACHEDWINNERDATASTORE Compute fixed training-time features once per jet.
    n = cfg.winnerImageSize;
    images = zeros(n,n,12,numel(jets),'single');
    features = zeros(numel(jets),4,'single');
    for j = 1:numel(jets)
        observation = winnerObservation(jets{j},normalization,cfg);
        images(:,:,:,j) = observation{1}; features(j,:) = observation{2};
    end
    targets = categorical(labels(:),[0 1],{'0','1'});
    batch = cfg.winnerBatchSize;
    ds = combine(arrayDatastore(images,IterationDimension=4,ReadSize=batch), ...
        arrayDatastore(features,ReadSize=batch),arrayDatastore(targets,ReadSize=batch));
end
