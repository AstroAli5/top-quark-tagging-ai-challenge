function [loss,gradients,Y] = modelLossGraphSAGE(parameters,X,A,numNodesPerGraph,T)
%MODELLOSSGRAPHSAGE Loss and gradients for one mini-batch, for use with
%dlfeval in the custom training loop (s4_train_graphsage.m).
%
%   [LOSS,GRADIENTS,Y] = MODELLOSSGRAPHSAGE(PARAMETERS,X,A,NUMNODESPERGRAPH,T)
%
%   T is the [numGraphs x 1] binary target (1 = top quark, 0 = background).
%   Uses binary cross-entropy via MATLAB's crossentropy function in
%   "multilabel" mode, which reduces to standard binary cross-entropy
%   for a single output column — the same loss-computation pattern
%   MathWorks uses in its documented GAT graph-classification example.

    Y = modelGraphSAGE(parameters,X,A,numNodesPerGraph);
    loss = crossentropy(Y,T,ClassificationMode="multilabel",DataFormat="BC");
    gradients = dlgradient(loss,parameters);
end
