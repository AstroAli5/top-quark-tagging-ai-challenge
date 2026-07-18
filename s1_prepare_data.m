% s1_prepare_data.m - REAL DATA 50k VERSION
clear; clc;
addpath('helpers')
fprintf("Step 1/7: Loading real jet data (50k)...\n");
dataFolder = '../data';
if ~isfolder(dataFolder); mkdir(dataFolder); end
raw = load('../data/jets_real_50k.mat');
particleData = double(raw.particleData);
labels = double(raw.labels(:));
numJets = size(particleData,1);
fprintf("Loaded %d jets. Signal: %d, Background: %d\n", numJets, sum(labels==1), sum(labels==0));
jetFourVectors = cell(numJets,1);
for j = 1:numJets
    thisJet = reshape(particleData(j,:),4,200)';
    isReal = any(thisJet ~= 0,2);
    jetFourVectors{j} = thisJet(isReal,:);
end
counts = cellfun(@(x)size(x,1),jetFourVectors);
fprintf("Particles per jet: min=%d median=%d max=%d\n",min(counts),round(median(counts)),max(counts));
save(fullfile(dataFolder,'jets_raw.mat'),'jetFourVectors','labels','-v7.3');
fprintf("Done. Saved 50k real jets to ../data/jets_raw.mat\n");
fprintf("NEXT STEP: run s2_build_representations.m\n");