function [tpr,fpr,auc] = computeROC(scores,labels)
%COMPUTEROC Binary ROC/AUC, with half credit for positive/negative ties.
%   Higher scores mean label 1. Both classes must be present.
%   Add an ROC point only after each complete group of equal scores.
    validateattributes(scores,{'numeric'},{'real','finite','vector','nonempty'});
    validateattributes(labels,{'numeric','logical'},{'real','finite','vector','nonempty'});
    scores = double(scores(:));
    labels = double(labels(:));
    if numel(scores) ~= numel(labels) || any(labels ~= 0 & labels ~= 1)
        error('topquark:InvalidLabels','Supply one binary label per score.');
    end
    numPos = sum(labels == 1);
    numNeg = sum(labels == 0);
    if numPos == 0 || numNeg == 0
        error('topquark:SingleClass','ROC/AUC requires both classes.');
    end
    metrics = rocmetrics(labels,scores,1);
    tpr = metrics.Metrics.TruePositiveRate;
    fpr = metrics.Metrics.FalsePositiveRate;
    % R2024a has rocmetrics but predates its separate auc method.
    auc = trapz(fpr,tpr);
end
