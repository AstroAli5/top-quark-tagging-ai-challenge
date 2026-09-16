function s6_robustness_test(cfg)
%S6_ROBUSTNESS_TEST Paired synthetic smearing; both models see identical jets.
if nargin < 1, cfg = projectConfig; end
fprintf('Step 6/7: Testing sensitivity to synthetic momentum smearing...\n');
split = load(fullfile(cfg.dataDir,'jet_split.mat'));
cnn = load(fullfile(cfg.modelsDir,'cnn_model.mat'));
sage = load(fullfile(cfg.modelsDir,'graphsage_model.mat'));
verifyDatasetIds(split,cnn,sage);
jetFourVectors = split.jetFourVectors;
labels = split.labels;
idxTest = split.idxTest;
imageSize = split.cfg.imageSize;
kNeighbors = split.cfg.kNeighbors;
noiseLevels = cfg.noiseLevels;
testFourVectors = jetFourVectors(idxTest);
testLabels = labels(idxTest);
numTest = numel(testFourVectors);

resultsDir = cfg.resultsDir;
if ~isfolder(resultsDir); mkdir(resultsDir); end

accCNNByNoise = zeros(size(noiseLevels));
aucCNNByNoise = zeros(size(noiseLevels));
accSAGEByNoise = zeros(size(noiseLevels));
aucSAGEByNoise = zeros(size(noiseLevels));

% Reset per level: the same standard-normal draws are scaled at each level.

for n = 1:numel(noiseLevels)
    noiseLevel = noiseLevels(n);
    rng(cfg.noiseSeed,'twister');
    fprintf("Noise level %.2f...\n", noiseLevel);

    noisyImages = zeros(imageSize,imageSize,1,numTest,"single");
    noisyNodeFeatures = cell(numTest,1);
    noisyAdjacency = cell(numTest,1);

    for j = 1:numTest
        noisyFV = injectDetectorNoise(testFourVectors{j},noiseLevel);
        noisyImages(:,:,1,j) = single(buildJetImage(noisyFV,imageSize));
        [nf,adj] = buildJetGraph(noisyFV,kNeighbors);
        noisyNodeFeatures{j} = single(nf);
        noisyAdjacency{j} = sparse(adj);
    end

    % --- CNN ---
    probCNN = predictCNN(cnn.netCNN,noisyImages,cnn.classNames,cfg.cnnBatchSize);
    predCNN = double(probCNN >= 0.5);
    accCNNByNoise(n) = mean(predCNN == testLabels);
    [~,~,aucCNNByNoise(n)] = computeROC(probCNN,testLabels);

    % --- GraphSAGE ---
    probSAGE = predictGraphSAGE(sage.parameters,noisyNodeFeatures,noisyAdjacency,cfg.graphBatchSize);
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
fig = figure('Visible','off');
cleanup = onCleanup(@() close(fig));
tiledlayout(1,2);

nexttile;
plot(noiseLevels,accCNNByNoise,"-o",LineWidth=1.5); hold on;
plot(noiseLevels,accSAGEByNoise,"-o",LineWidth=1.5);
xlabel("Synthetic component smearing (fractional)"); ylabel("Accuracy");
legend("CNN","GraphSAGE",Location="southwest");
title("Accuracy vs synthetic smearing"); grid on;

nexttile;
plot(noiseLevels,aucCNNByNoise,"-o",LineWidth=1.5); hold on;
plot(noiseLevels,aucSAGEByNoise,"-o",LineWidth=1.5);
xlabel("Synthetic component smearing (fractional)"); ylabel("AUC");
legend("CNN","GraphSAGE",Location="southwest");
title("AUC vs synthetic smearing"); grid on;

saveas(fig,fullfile(resultsDir,"robustness_curves.png"));

fprintf("\nSaved results/robustness_results.csv and results/robustness_curves.png\n");
end
