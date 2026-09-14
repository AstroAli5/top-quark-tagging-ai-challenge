function probabilities = predictGraphSAGE(parameters,features,adjacency,batchSize)
%PREDICTGRAPHSAGE Bounded-memory inference, preserving input jet order.
    if nargin < 4, batchSize = 64; end
    validateattributes(batchSize,{'numeric'},{'scalar','integer','positive'});
    if numel(features) ~= numel(adjacency)
        error('topquark:InvalidBatch','Feature and adjacency counts differ.');
    end
    probabilities = zeros(numel(features),1);
    for first = 1:batchSize:numel(features)
        idx = first:min(first+batchSize-1,numel(features));
        [X,A,numNodes] = preprocessGraphMiniBatch(features(idx),adjacency(idx));
        Y = modelGraphSAGE(parameters,dlarray(X),A,numNodes);
        probabilities(idx) = double(extractdata(Y));
    end
    if any(~isfinite(probabilities) | probabilities < 0 | probabilities > 1)
        error('topquark:InvalidProbabilities','GraphSAGE returned invalid probabilities.');
    end
end
