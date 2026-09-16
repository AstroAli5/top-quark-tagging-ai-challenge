function weights = initializeGlorot(sz,numOut,numIn,className)
%INITIALIZEGLOROT Glorot (Xavier) weight initialization.
%
%   WEIGHTS = INITIALIZEGLOROT(SZ,NUMOUT,NUMIN) returns an array of size
%   SZ containing weights drawn from a uniform distribution scaled by the
%   number of input (NUMIN) and output (NUMOUT) connections, returned as
%   a dlarray. This keeps the scale of gradients roughly the same across
%   every layer, which helps training converge.
%
%   This follows the same initialization scheme used in MathWorks' own
%   documented graph neural network examples (Node Classification Using
%   GCN; Multilabel Graph Classification Using GAT).

    arguments
        sz
        numOut
        numIn
        className = 'single'
    end

    Z = 2*rand(sz,className) - 1;
    bound = sqrt(6/(numIn + numOut));
    weights = bound*Z;
    weights = dlarray(weights);
end
