function [features,adjacency,numNodes,target] = preprocessGraphMiniBatch(featureData,adjacencyData,targetData)
%PREPROCESSGRAPHMINIBATCH Merge a mini-batch of separate jet graphs into
%one combined block-diagonal graph for training.
%
%   [FEATURES,ADJACENCY,NUMNODES,TARGET] = PREPROCESSGRAPHMINIBATCH(FEATUREDATA,ADJACENCYDATA,TARGETDATA)
%
%   FEATUREDATA, ADJACENCYDATA - cell arrays, one cell per jet in the
%       mini-batch. Each jet can have a different number of particles —
%       no padding is needed, since every jet is already stored at its
%       own natural size (unlike a fixed-size image).
%   TARGETDATA - cell array or vector of 0/1 labels, one per jet
%       (optional — omit to preprocess data for prediction only).
%
%   Follows the same block-diagonal-adjacency merging pattern MathWorks
%   uses in its documented "Multilabel Graph Classification Using GAT"
%   example, adapted for variable-size (rather than zero-padded) graphs
%   and without adding self-loops (this project's GraphSAGE layer keeps
%   self and neighbor information separate — see helpers/graphSAGELayer.m).

    numGraphs = numel(featureData);
    features = [];
    adjacency = sparse([]);
    numNodes = zeros(numGraphs,1);

    for i = 1:numGraphs
        thisFeatures = featureData{i};
        thisAdjacency = adjacencyData{i};

        numNodes(i) = size(thisFeatures,1);
        features = [features; thisFeatures]; %#ok<AGROW>
        adjacency = blkdiag(adjacency,thisAdjacency);
    end

    if nargin > 2 && ~isempty(targetData)
        if iscell(targetData)
            target = cell2mat(targetData(:));
        else
            target = targetData(:);
        end
    else
        target = [];
    end
end
