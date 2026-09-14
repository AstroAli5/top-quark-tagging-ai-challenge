function [features,adjacency,numNodes,target] = preprocessGraphMiniBatch(featureData,adjacencyData,targetData)
%PREPROCESSGRAPHMINIBATCH Stack graphs as a sparse block-diagonal batch.
%   Double CPU arrays match sparse adjacency and GraphSAGE parameters.
    if isempty(featureData) || numel(featureData) ~= numel(adjacencyData)
        error('topquark:InvalidBatch','Supply matching nonempty graph cells.');
    end
    numGraphs = numel(featureData);
    numNodes = cellfun(@(x) size(x,1),featureData(:));
    if any(numNodes == 0)
        error('topquark:EmptyGraph','A batch cannot contain an empty graph.');
    end
    blocks = cell(numGraphs,1);
    for i = 1:numGraphs
        validateattributes(featureData{i},{'numeric'},{'2d','real','finite'});
        if ~isequal(size(adjacencyData{i}),[numNodes(i) numNodes(i)])
            error('topquark:InvalidAdjacency','Adjacency must match the node count.');
        end
        blocks{i} = sparse(double(adjacencyData{i}));
    end
    features = double(vertcat(featureData{:}));
    adjacency = blkdiag(blocks{:});
    target = [];
    if nargin > 2 && ~isempty(targetData)
        if iscell(targetData), targetData = cell2mat(targetData(:)); end
        target = double(targetData(:));
        if numel(target) ~= numGraphs || any(target ~= 0 & target ~= 1)
            error('topquark:InvalidLabels','Supply one binary target per graph.');
        end
    end
end
