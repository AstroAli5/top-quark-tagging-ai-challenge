function parameters = initializeGraphSAGE(numFeatures,hiddenSize)
%INITIALIZEGRAPHSAGE Double precision matches the sparse CPU adjacency.
    parameters.sage1.Weights = initializeGlorot( ...
        [2*numFeatures hiddenSize],hiddenSize,2*numFeatures,'double');
    parameters.sage2.Weights = initializeGlorot( ...
        [2*hiddenSize hiddenSize],hiddenSize,2*hiddenSize,'double');
    parameters.sage3.Weights = initializeGlorot( ...
        [2*hiddenSize hiddenSize],hiddenSize,2*hiddenSize,'double');
    parameters.classify.Weights = initializeGlorot([hiddenSize 1],1,hiddenSize,'double');
end
