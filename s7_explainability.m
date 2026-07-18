% s7_explainability.m
% -------------------------------------------------------------------------
% STEP 7 of 7 — STRETCH GOAL: lightweight explainability.
%
% Two simple, well-understood techniques, rather than a heavyweight
% dedicated GNN-explainability framework (deliberately — see README for
% why this was scoped down from the original plan):
%
%   1. Permutation feature importance for GraphSAGE: shuffle one node
%      feature at a time across the test set and measure how much AUC
%      drops. A bigger drop means the model relies on that feature more.
%   2. Radial occlusion for the CNN: zero out the jet image outside
%      shrinking radii from the jet axis and measure how much AUC drops,
%      to see how spatially concentrated the CNN's decision is.
%
% Toolboxes required: Deep Learning Toolbox.
% -------------------------------------------------------------------------

clear; clc;
addpath('/MATLAB Drive/src/helpers');

fprintf("Step 7/7: Explainability (stretch)...\n");

resultsDir = '../results';
if ~isfolder(resultsDir); mkdir(resultsDir); end

%% 1. GraphSAGE: permutation feature importance
load('../data/jet_graphs.mat',"jetNodeFeatures","jetAdjacency","labels","idxTest");
load('../models/graphsage_model.mat',"parameters");

featureNames = ["deltaEta","deltaPhi","log(pT)","log(E)"];
[XBase,ABase,numNodesBase] = preprocessGraphMiniBatch(jetNodeFeatures(idxTest),jetAdjacency(idxTest));
labelsTest = labels(idxTest);

probBase = extractdata(modelGraphSAGE(parameters,dlarray(XBase),ABase,numNodesBase));
[~,~,aucBase] = computeROC(probBase,labelsTest);
fprintf("Baseline AUC (nothing shuffled): %.4f\n", aucBase);

rng(11);
aucDropByFeature = zeros(1,4);
for f = 1:4
    XPerturbed = XBase;
    XPerturbed(:,f) = XPerturbed(randperm(size(XPerturbed,1)),f); % shuffle this feature only
    probPerturbed = extractdata(modelGraphSAGE(parameters,dlarray(XPerturbed),ABase,numNodesBase));
    [~,~,aucPerturbed] = computeROC(probPerturbed,labelsTest);
    aucDropByFeature(f) = aucBase - aucPerturbed;
    fprintf("Shuffling %-10s : AUC drops to %.4f (drop = %.4f)\n", ...
        featureNames(f), aucPerturbed, aucDropByFeature(f));
end

figure;
bar(categorical(featureNames),aucDropByFeature);
ylabel("AUC drop when shuffled"); title("GraphSAGE feature importance");
grid on;
saveas(gcf,fullfile(resultsDir,"graphsage_feature_importance.png"));

importanceTable = table(featureNames',aucDropByFeature',VariableNames={'Feature','AUCDrop'});
writetable(importanceTable,fullfile(resultsDir,"graphsage_feature_importance.csv"));

%% 2. CNN: radial occlusion
load('../data/jet_images.mat',"jetImages","labels","idxTest");
load('../models/cnn_model.mat',"netCNN");

XImgTest = jetImages(:,:,:,idxTest);
labelsImgTest = labels(idxTest);

imageSize = size(XImgTest,1);
[gridX,gridY] = meshgrid(1:imageSize,1:imageSize);
centerXY = (imageSize+1)/2;
radiusMap = sqrt((gridX-centerXY).^2 + (gridY-centerXY).^2);
maxRadius = max(radiusMap(:));

radiusFractions = [1.0, 0.75, 0.5, 0.35, 0.2];
aucByRadius = zeros(size(radiusFractions));

for r = 1:numel(radiusFractions)
    keepRadius = radiusFractions(r)*maxRadius;
    mask = radiusMap <= keepRadius;

    XMasked = XImgTest .* single(mask);
    scores = predict(netCNN,XMasked);
    probMasked = scores(:,2);
    [~,~,aucByRadius(r)] = computeROC(probMasked,labelsImgTest);

    fprintf("Keeping central %.0f%% radius: AUC = %.4f\n", 100*radiusFractions(r), aucByRadius(r));
end

figure;
plot(radiusFractions,aucByRadius,"-o",LineWidth=1.5);
xlabel("Fraction of image radius kept (centered on jet axis)"); ylabel("AUC");
title("CNN: how much of the jet image actually matters?");
grid on; set(gca,"XDir","reverse");
saveas(gcf,fullfile(resultsDir,"cnn_radial_occlusion.png"));

radiusTable = table(radiusFractions',aucByRadius',VariableNames={'RadiusFractionKept','AUC'});
writetable(radiusTable,fullfile(resultsDir,"cnn_radial_occlusion.csv"));

fprintf("\nAll 7 steps complete. See README.md Results section - fill in your real numbers.\n");
