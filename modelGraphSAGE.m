function Y = modelGraphSAGE(parameters,X,A,numNodesPerGraph)
%MODELGRAPHSAGE Full GraphSAGE forward pass for jet (graph) classification.
%
%   Y = MODELGRAPHSAGE(PARAMETERS,X,A,NUMNODESPERGRAPH)
%
%   Stacks 3 GraphSAGE aggregation layers, pools each jet's particle
%   (node) embeddings into one embedding per jet, then applies a
%   classifier head. Returns the predicted probability that each jet in
%   the mini-batch is a top-quark jet (as opposed to background).
%
%   X                - [totalNodes x numFeatures] stacked node features
%   A                - [totalNodes x totalNodes] block-diagonal adjacency
%                      matrix (see helpers/buildJetGraph.m and the
%                      mini-batch preprocessing in s4_train_graphsage.m)
%   NUMNODESPERGRAPH  - number of particles in each jet in the mini-batch

    Z1 = X;
    Z2 = graphSAGELayer(Z1,A,parameters.sage1);
    Z3 = graphSAGELayer(Z2,A,parameters.sage2);
    Z4 = graphSAGELayer(Z3,A,parameters.sage3);

    graphEmbedding = globalMeanPool(Z4,numNodesPerGraph);

    logits = graphEmbedding * parameters.classify.Weights;
    Y = sigmoid(logits);
end
