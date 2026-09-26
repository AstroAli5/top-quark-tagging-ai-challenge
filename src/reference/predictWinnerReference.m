function probabilities = predictWinnerReference(model,jets,batchSize,executionEnvironment)
%PREDICTWINNERREFERENCE Bounded inference using saved training preprocessing.
    if nargin < 3, batchSize = model.cfg.winnerBatchSize; end
    if nargin < 4, executionEnvironment = model.cfg.executionEnvironment; end
    executionEnvironment = validatestring(executionEnvironment,{'cpu','gpu','auto'});
    useGPU = strcmp(executionEnvironment,'gpu');
    if strcmp(executionEnvironment,'auto')
        useGPU = exist('canUseGPU','file') && canUseGPU;
    end
    net = model.netWinner;
    if useGPU, net = dlupdate(@gpuArray,net); else, net = dlupdate(@gather,net); end
    validateattributes(batchSize,{'numeric'},{'scalar','integer','positive'});
    positive = find(string(model.classNames) == "1");
    assert(isscalar(positive) && numel(model.classNames) == 2,'Invalid class map.');
    probabilities = zeros(numel(jets),1);
    n = model.cfg.winnerImageSize;
    for start = 1:batchSize:numel(jets)
        indices = start:min(start+batchSize-1,numel(jets));
        images = zeros(n,n,12,numel(indices),'single');
        features = zeros(4,numel(indices),'single');
        for j = 1:numel(indices)
            observation = winnerObservation(jets{indices(j)},model.normalization,model.cfg);
            images(:,:,:,j) = observation{1};
            features(:,j) = observation{2}.';
        end
        if useGPU, images = gpuArray(images); features = gpuArray(features); end
        scores = predict(net,dlarray(images,'SSCB'),dlarray(features,'CB'));
        scores = double(gather(extractdata(scores)));
        scores = reshape(scores,2,[]).';
        if any(~isfinite(scores(:))) || any(scores(:) < 0 | scores(:) > 1) || ...
                any(abs(sum(scores,2)-1) > 1e-5)
            error('topquark:InvalidProbabilities','Invalid winner-reference probabilities.');
        end
        probabilities(indices) = scores(:,positive);
    end
end
