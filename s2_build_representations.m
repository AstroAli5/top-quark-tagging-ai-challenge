% s2_build_representations.m
% -------------------------------------------------------------------------
% STEP 2 of 7 — Turn the raw per-jet particle four-vectors from s1 into
% the two representations the two models need:
%   - a 2D "jet image" for the CNN baseline
%   - a graph (node features + adjacency matrix) for the GraphSAGE model
%
% Also splits into train/validation/test partitions here, once, so both
% models are compared on exactly the same jets later.
%
% Toolboxes required: none beyond base MATLAB.
% -------------------------------------------------------------------------

clear; clc;
addpath('/MATLAB Drive/src/helpers');

fprintf("Step 2/7: Building jet images and jet graphs...\n");

load('/MATLAB Drive/data/jets_raw.mat',"jetFourVectors","labels");
numJets = numel(jetFourVectors);

% Drop any jet with fewer than 3 particles — too little structure to
% build a meaningful graph or image from, and vanishingly rare in this
% dataset.
validJets = cellfun(@(x) size(x,1), jetFourVectors) >= 3;
jetFourVectors = jetFourVectors(validJets);
labels = labels(validJets);
numJets = numel(jetFourVectors);
fprintf("Using %d jets after dropping near-empty ones.\n", numJets);

%% 1. Build jet images (for the CNN)
imageSize = 32;
jetImages = zeros(imageSize,imageSize,1,numJets,"single");

fprintf("Building jet images...\n");
for j = 1:numJets
    jetImages(:,:,1,j) = single(buildJetImage(jetFourVectors{j},imageSize));
    if mod(j,2000) == 0 || j == numJets
        fprintf("  %d / %d\n", j, numJets);
    end
end

%% 2. Build jet graphs (for GraphSAGE)
kNeighbors = 6;
jetNodeFeatures = cell(numJets,1);
jetAdjacency = cell(numJets,1);

fprintf("Building jet graphs (k=%d nearest neighbors)...\n", kNeighbors);
for j = 1:numJets
    [nf,adj] = buildJetGraph(jetFourVectors{j},kNeighbors);
    jetNodeFeatures{j} = single(nf);
    jetAdjacency{j} = adj;
    if mod(j,2000) == 0 || j == numJets
        fprintf("  %d / %d\n", j, numJets);
    end
end

%% 3. Train / validation / test split (70% / 15% / 15%), stratified is
%     unnecessary here since the dataset is close to balanced already.
rng(42);
if ~isfolder('/MATLAB Drive/models'); mkdir('/MATLAB Drive/models'); end
if ~isfolder('/MATLAB Drive/results'); mkdir('/MATLAB Drive/results'); end % fixed seed so results are reproducible run to run
idxShuffled = randperm(numJets);

nTrain = round(0.70*numJets);
nVal   = round(0.15*numJets);

idxTrain = idxShuffled(1:nTrain);
idxVal   = idxShuffled(nTrain+1:nTrain+nVal);
idxTest  = idxShuffled(nTrain+nVal+1:end);

fprintf("Split: %d train, %d validation, %d test.\n", numel(idxTrain), numel(idxVal), numel(idxTest));

%% 4. Save both representations, split, ready for training
modelsDir = '/MATLAB Drive/data';

save(fullfile(modelsDir,"jet_images.mat"), ...
    "jetImages","labels","idxTrain","idxVal","idxTest","-v7.3");

save(fullfile(modelsDir,"jet_graphs.mat"), ...
    "jetNodeFeatures","jetAdjacency","labels","idxTrain","idxVal","idxTest","-v7.3");

% Also keep the raw four-vectors alongside the split indices — the
% robustness test in s6 needs to re-build graphs/images from noise-
% injected four-vectors later.
save(fullfile(modelsDir,"jet_split.mat"), ...
    "jetFourVectors","labels","idxTrain","idxVal","idxTest","-v7.3");

fprintf("\nDone. Saved jet_images.mat, jet_graphs.mat, and jet_split.mat to data/\n");
fprintf("NEXT STEP: run s3_train_cnn.m and s4_train_graphsage.m\n");
