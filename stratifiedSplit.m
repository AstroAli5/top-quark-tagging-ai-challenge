function [idxTrain,idxVal,idxTest] = stratifiedSplit(labels,seed)
%STRATIFIEDSPLIT Reproducible 70/15/15 split, with both classes in each part.
    labels = double(labels(:));
    if any(labels ~= 0 & labels ~= 1)
        error('topquark:InvalidLabels','Expected binary labels.');
    end
    previous = rng;
    cleanup = onCleanup(@() rng(previous));
    rng(seed,'twister');
    idxTrain = []; idxVal = []; idxTest = [];
    for label = [0 1]
        indices = find(labels == label);
        n = numel(indices);
        if n < 3
            error('topquark:TooFewJets','Need at least three usable jets per class.');
        end
        indices = indices(randperm(n));
        nTrain = min(n-2,max(1,round(0.70*n)));
        nVal = min(n-nTrain-1,max(1,round(0.15*n)));
        idxTrain = [idxTrain; indices(1:nTrain)]; %#ok<AGROW>
        idxVal = [idxVal; indices(nTrain+1:nTrain+nVal)]; %#ok<AGROW>
        idxTest = [idxTest; indices(nTrain+nVal+1:end)]; %#ok<AGROW>
    end
    idxTrain = idxTrain(randperm(numel(idxTrain)));
    idxVal = idxVal(randperm(numel(idxVal)));
    idxTest = idxTest(randperm(numel(idxTest)));
end
