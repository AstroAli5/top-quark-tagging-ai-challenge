function [tpr,fpr,auc] = computeROC(scores,labels)
%COMPUTEROC Compute a ROC curve and its AUC for binary classification
%scores, without needing Statistics and Machine Learning Toolbox's
%perfcurve.
%
%   [TPR,FPR,AUC] = COMPUTEROC(SCORES,LABELS)
%
%   SCORES - predicted probability/score per example (higher = more
%            likely to be class 1 = top quark).
%   LABELS - true 0/1 label per example.
%
%   Standard approach: sort examples by score, then walk down the sorted
%   list accumulating true/false positives. AUC is measured by
%   integrating the resulting curve with trapz.

    scores = scores(:);
    labels = labels(:);

    [~,order] = sort(scores,"descend");
    sortedLabels = labels(order);

    numPos = sum(labels == 1);
    numNeg = sum(labels == 0);

    tpr = [0; cumsum(sortedLabels == 1)/max(numPos,1)];
    fpr = [0; cumsum(sortedLabels == 0)/max(numNeg,1)];

    auc = trapz(fpr,tpr);
end
