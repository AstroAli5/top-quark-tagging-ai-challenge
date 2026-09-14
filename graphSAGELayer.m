function Znext = graphSAGELayer(Z,A,weights)
%GRAPHSAGELAYER One GraphSAGE "mean aggregator" layer.
%
%   ZNEXT = GRAPHSAGELAYER(Z,A,WEIGHTS)
%
%   Z       - [numNodes x numFeatures] current node feature matrix
%   A       - [numNodes x numNodes] adjacency matrix (block-diagonal
%             across the mini-batch, NO self-loops — this function keeps
%             "self" and "neighbor" information separate, unlike a GCN)
%   WEIGHTS - struct with field Weights, size
%             [2*numFeatures x numOutputFeatures]
%
%   Implements the GraphSAGE mean-aggregator update rule (Hamilton,
%   Ying & Leskovec, 2017):
%
%       h_N(v)  = mean of h_u over neighbors u of v
%       h_v_new = ReLU( W * [h_v , h_N(v)] )
%       h_v_new = h_v_new / norm(h_v_new)      (row L2-normalize)
%
%   The neighbor-averaging step is implemented as a single normalized
%   adjacency-matrix multiply, the same style of operation MathWorks uses
%   in its documented GCN example (there with symmetric normalization;
%   here with simple row-mean normalization, since GraphSAGE keeps a
%   node's own features and its neighbors' features separate instead of
%   blending them together).

    % Row-normalize the adjacency matrix so each node's neighbor features
    % are AVERAGED, not summed (guard against isolated nodes with no
    % neighbors by flooring the degree at 1).
    degree = max(full(sum(A,2)),1);
    AMean = spdiags(1./degree,0,size(A,1),size(A,1)) * A;

    % MATLAB R2024a dispatches a sparse 1-by-1 multiply as scalar
    % multiplication, which dlarray rejects. Densify only this scalar.
    if isscalar(AMean), AMean = full(AMean); end
    neighborAgg = AMean * Z;

    combined = [Z, neighborAgg];
    Znext = relu(combined * weights.Weights);

    % L2-normalize each node's feature vector, as in the original
    % GraphSAGE paper.
    rowNorm = sqrt(sum(Znext.^2,2) + 1e-8);
    Znext = Znext ./ rowNorm;
end
