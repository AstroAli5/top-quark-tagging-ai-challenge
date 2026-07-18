function graphFeatures = globalMeanPool(nodeFeatures,numNodesPerGraph)
%GLOBALMEANPOOL Turn per-particle (node) features into one feature vector
%per jet (graph) by averaging.
%
%   GRAPHFEATURES = GLOBALMEANPOOL(NODEFEATURES,NUMNODESPERGRAPH)
%
%   NODEFEATURES is a stacked [totalNodes x numFeatures] matrix — every
%   particle from every jet in the mini-batch, one after another.
%   NUMNODESPERGRAPH lists how many particles belong to each jet, in the
%   same order they were stacked, so this function knows where one jet
%   ends and the next begins.
%
%   This is the "readout" step that turns particle-level information into
%   a single jet-level embedding for classification (see
%   docs/CONCEPTS.md). The approach — average node features per graph
%   using the graph boundaries — follows the same pattern as the
%   globalAveragePool function in MathWorks' own documented "Multilabel
%   Graph Classification Using GAT" example.

    numGraphs = numel(numNodesPerGraph);
    numFeatures = size(nodeFeatures,2);
    graphFeatures = zeros(numGraphs,numFeatures,"like",nodeFeatures);

    startIdx = 1;
    for i = 1:numGraphs
        endIdx = startIdx + numNodesPerGraph(i) - 1;
        graphFeatures(i,:) = mean(nodeFeatures(startIdx:endIdx,:),1);
        startIdx = endIdx + 1;
    end
end
