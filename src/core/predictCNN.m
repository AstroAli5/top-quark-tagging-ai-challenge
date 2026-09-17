function probabilities = predictCNN(netCNN,images,classNames,batchSize)
%PREDICTCNN Return P(signal=1) with explicit batch/channel layout and class map.
    if nargin < 4, batchSize = 128; end
    positiveColumn = find(string(classNames) == "1");
    if numel(positiveColumn) ~= 1 || numel(classNames) ~= 2
        error('topquark:InvalidClasses','Expected class names 0 and 1.');
    end
    scores = minibatchpredict(netCNN,images,MiniBatchSize=batchSize, ...
        ExecutionEnvironment='cpu',InputDataFormats='SSCB',OutputDataFormats='BC');
    if isa(scores,'dlarray'), scores = extractdata(scores); end
    scores = double(scores);
    if size(scores,1) ~= size(images,4) || size(scores,2) ~= 2 || ...
            any(~isfinite(scores(:))) || any(scores(:) < 0 | scores(:) > 1) || ...
            any(abs(sum(scores,2)-1) > 1e-5)
        error('topquark:InvalidProbabilities', ...
            'Expected two normalized class probabilities per jet. Retrain with softmax.');
    end
    probabilities = scores(:,positiveColumn);
end
