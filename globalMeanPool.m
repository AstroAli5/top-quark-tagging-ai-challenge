function graphFeatures = globalMeanPool(nodeFeatures,numNodesPerGraph)
%GLOBALMEANPOOL Average nodes per graph without breaking autodifferentiation.
    counts = double(numNodesPerGraph(:));
    if isempty(counts) || any(counts < 1 | counts ~= floor(counts)) || ...
            sum(counts) ~= size(nodeFeatures,1)
        error('topquark:InvalidNodeCounts','Node counts must partition all feature rows.');
    end
    pooled = cell(numel(counts),1);
    first = 1;
    for i = 1:numel(counts)
        last = first + counts(i) - 1;
        pooled{i} = mean(nodeFeatures(first:last,:),1);
        first = last + 1;
    end
    graphFeatures = cat(1,pooled{:});
end
