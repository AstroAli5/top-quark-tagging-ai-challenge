% s5_evaluate_baseline.m
% -------------------------------------------------------------------------
% STEP 5 of 7 — Evaluate both trained models on the clean held-out test
% set: accuracy, AUC, and an ROC curve comparing them directly.
%
% Toolboxes required: Deep Learning Toolbox.
% -------------------------------------------------------------------------

clear; clc;
addpath('/MATLAB Drive/src/helpers');

fprintf("Step 5/7: Evaluating both models on the clean test set...\n");

resultsDir = '/MATLAB Drive/results';
if ~isfolder(resultsDir); mkdir(resultsDir); end

%% CNN evaluation
load('/MATLAB Drive/data/jet_images.mat',"jetImages","labels","idxTest");
load('/MATLAB Drive/models/cnn_model.mat',"netCNN");

XTest = jetImages(:,:,:,idxTest);
labelsTest = labels(idxTest);

scoresCNN = predict(netCNN,XTest);
probCNN = scoresCNN(:,2);              % P(class = 1 = top quark)
predCNN = double(probCNN >= 0.5);

accCNN = mean(predCNN == labelsTest);
[tprCNN,fprCNN,aucCNN] = computeROC(probCNN,labelsTest);

fprintf("CNN         : accuracy %.4f, AUC %.4f\n", accCNN, aucCNN);

%% GraphSAGE evaluation
load('/MATLAB Drive/data/jet_graphs.mat',"jetNodeFeatures","jetAdjacency","labels","idxTest");
load('/MATLAB Drive/models/graphsage_model.mat',"parameters");

[XTestGraph,ATestGraph,numNodesTest] = preprocessGraphMiniBatch(jetNodeFeatures(idxTest),jetAdjacency(idxTest));
probSAGE = extractdata(modelGraphSAGE(parameters,dlarray(XTestGraph),ATestGraph,numNodesTest));
predSAGE = double(probSAGE >= 0.5);
labelsTestGraph = labels(idxTest);

accSAGE = mean(predSAGE == labelsTestGraph);
[tprSAGE,fprSAGE,aucSAGE] = computeROC(probSAGE,labelsTestGraph);

fprintf("GraphSAGE   : accuracy %.4f, AUC %.4f\n", accSAGE, aucSAGE);

%% Save comparison table
resultsTable = table(["CNN";"GraphSAGE"],[accCNN;accSAGE],[aucCNN;aucSAGE], ...
    'VariableNames',{'Model','Accuracy','AUC'});
writetable(resultsTable,fullfile(resultsDir,"baseline_comparison.csv"));
disp(resultsTable);

%% ROC plot
figure;
plot(fprCNN,tprCNN,"LineWidth",1.5); hold on;
plot(fprSAGE,tprSAGE,"LineWidth",1.5);
plot([0 1],[0 1],"k--");
xlabel("False Positive Rate"); ylabel("True Positive Rate");
legend("CNN (AUC=" + round(aucCNN,3) + ")", ...
       "GraphSAGE (AUC=" + round(aucSAGE,3) + ")", ...
       "Random",Location="southeast");
title("ROC: CNN vs GraphSAGE, clean test set");
grid on;
saveas(gcf,fullfile(resultsDir,"roc_baseline.png"));

fprintf("\nSaved results/baseline_comparison.csv and results/roc_baseline.png\n");
fprintf("NEXT STEP: run s6_robustness_test.m\n");
