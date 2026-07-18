% s6_robustness_test.m
% -------------------------------------------------------------------------
% STEP 6 of 7 — MAIN ORIGINAL CONTRIBUTION.
%
% Both models are trained once on clean simulated data (s3, s4). Real
% detectors are never perfectly precise. This script asks the question
% the reference material never checks: how much does each model's
% performance degrade as measurements get noisier, and does the CNN or
% GraphSAGE degrade more gracefully?
%
% Toolboxes required: Deep Learning Toolbox.
% -------------------------------------------------------------------------

clear; clc;
addpath('/MATLAB Drive/src/helpers');

fprintf("Step 6/7: Robustness test - accuracy vs detector noise level...\n");

load('../data/jet_split.mat',"jetFourVectors","labels","idxTest");
load('../models/cnn_model.mat',"netCNN");
load('../models/graphsage_model.mat',"parameters");

noiseLevels = [0, 0.02, 0.05, 0.10, 0.20, 0.35];
testFourVectors = jetFourVectors(idxTest);
testLabels = labels(idxTest);
numTest = numel(testFourVectors);

resultsDir = '../results';
if ~isfolder(resultsDir); mkdir(resultsDir); end

accCNNByNoise = zeros(size(noiseLevels));
aucCNNByNoise = zeros(size(noiseLevels));
accSAGEByNoise = zeros(size(noiseLevels));
aucSAGEByNoise = zeros(size(noiseLevels));

rng(7); % fixed seed so noise levels are comparable run to run

for n = 1:numel(noiseLevels)
    noiseLevel = noiseLevels(n);
    fprintf("Noise level %.2f...\n", noiseLevel);

    noisyImages = zeros(32,32,1,numTest,"single");
    noisyNodeFeatures = cell(numTest,1);
    noisyAdjacency = cell(numTest,1);

    for j = 1:numTest
        noisyFV = injectDetectorNoise(testFourVectors{j},noiseLevel);
        noisyImages(:,:,1,j) = single(buildJetImage(noisyFV,32));
        [nf,adj] = buildJetGraph(noisyFV,6);
        noisyNodeFeatures{j} = single(nf);
        noisyAdjacency{j} = adj;
    end

    % --- CNN ---
    scoresCNN = predict(netCNN,noisyImages);
    probCNN = scoresCNN(:,2);
    predCNN = double(probCNN >= 0.5);
    accCNNByNoise(n) = mean(predCNN == testLabels);
    [~,~,aucCNNByNoise(n)] = computeROC(probCNN,testLabels);

    % --- GraphSAGE ---
    [XGraph,AGraph,numNodesGraph] = preprocessGraphMiniBatch(noisyNodeFeatures,noisyAdjacency);
    probSAGE = extractdata(modelGraphSAGE(parameters,dlarray(XGraph),AGraph,numNodesGraph));
    predSAGE = double(probSAGE >= 0.5);
    accSAGEByNoise(n) = mean(predSAGE == testLabels);
    [~,~,aucSAGEByNoise(n)] = computeROC(probSAGE,testLabels);

    fprintf("  CNN: acc %.4f, AUC %.4f | GraphSAGE: acc %.4f, AUC %.4f\n", ...
        accCNNByNoise(n), aucCNNByNoise(n), accSAGEByNoise(n), aucSAGEByNoise(n));
end

%% Save results
robustnessTable = table(noiseLevels',accCNNByNoise',aucCNNByNoise',accSAGEByNoise',aucSAGEByNoise', ...
    'VariableNames',{'NoiseLevel','CNN_Accuracy','CNN_AUC','GraphSAGE_Accuracy','GraphSAGE_AUC'});
writetable(robustnessTable,fullfile(resultsDir,"robustness_results.csv"));
disp(robustnessTable);

%% Plot degradation curves
figure;
tiledlayout(1,2);

nexttile;
plot(noiseLevels,accCNNByNoise,"-o",LineWidth=1.5); hold on;
plot(noiseLevels,accSAGEByNoise,"-o",LineWidth=1.5);
xlabel("Detector noise level (fractional)"); ylabel("Accuracy");
legend("CNN","GraphSAGE",Location="southwest");
title("Accuracy vs detector noise"); grid on;

nexttile;
plot(noiseLevels,aucCNNByNoise,"-o",LineWidth=1.5); hold on;
plot(noiseLevels,aucSAGEByNoise,"-o",LineWidth=1.5);
xlabel("Detector noise level (fractional)"); ylabel("AUC");
legend("CNN","GraphSAGE",Location="southwest");
title("AUC vs detector noise"); grid on;

saveas(gcf,fullfile(resultsDir,"robustness_curves.png"));

fprintf("\nSaved results/robustness_results.csv and results/robustness_curves.png\n");
fprintf("NEXT STEP: run s7_explainability.m (stretch goal), then write up your results.\n");
