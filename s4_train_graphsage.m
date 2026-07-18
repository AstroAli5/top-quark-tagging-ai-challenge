% s4_train_graphsage.m
% -------------------------------------------------------------------------
% STEP 4 of 7 — Train the GraphSAGE model on the jet graphs, using a
% custom training loop (graph data is irregular/sparse, so — as in
% MathWorks' own documented GCN and GAT examples — a custom training loop
% is the right tool rather than trainnet).
%
% This is the least conventional part of this project. I built it by
% closely following the exact patterns MathWorks documents in "Node
% Classification Using Graph Convolutional Network" and "Multilabel
% Graph Classification Using Graph Attention Networks" (block-diagonal
% batching, minibatchqueue, dlfeval + adamupdate), swapping in
% GraphSAGE's specific neighbor-mean-aggregation update rule instead of
% GCN's symmetric-normalized convolution or GAT's attention mechanism.
% I have not been able to execute this myself, so budget extra time here
% for debugging — that's expected, not a sign anything is wrong.
%
% Toolboxes required: Deep Learning Toolbox.
% -------------------------------------------------------------------------

clear; clc;
addpath('/MATLAB Drive/src/helpers');

fprintf("Step 4/7: Training GraphSAGE model...\n");

load('../data/jet_graphs.mat',"jetNodeFeatures","jetAdjacency","labels","idxTrain","idxVal","idxTest");

numFeatures = size(jetNodeFeatures{1},2); % 4: dEta, dPhi, log(pT), log(E)
hiddenSize = 32;
numClasses = 1; % binary: sigmoid output

%% 1. Initialize parameters
parameters = struct;

parameters.sage1.Weights = initializeGlorot([2*numFeatures,hiddenSize],hiddenSize,2*numFeatures);
parameters.sage2.Weights = initializeGlorot([2*hiddenSize,hiddenSize],hiddenSize,2*hiddenSize);
parameters.sage3.Weights = initializeGlorot([2*hiddenSize,hiddenSize],hiddenSize,2*hiddenSize);
parameters.classify.Weights = initializeGlorot([hiddenSize,numClasses],numClasses,hiddenSize);

%% 2. Build datastores for mini-batch training
featuresTrainDs  = arrayDatastore(jetNodeFeatures(idxTrain),OutputType="same",ReadSize=1);
adjacencyTrainDs = arrayDatastore(jetAdjacency(idxTrain),OutputType="same",ReadSize=1);
targetTrainDs    = arrayDatastore(labels(idxTrain));

dsTrain = combine(featuresTrainDs,adjacencyTrainDs,targetTrainDs);

miniBatchSize = 64;
mbq = minibatchqueue(dsTrain,4, ...
    MiniBatchSize=miniBatchSize, ...
    PartialMiniBatch="discard", ...
    MiniBatchFcn=@preprocessGraphMiniBatch, ...
    OutputCast="double", ...
    OutputAsDlarray=[1 0 0 0], ...
    OutputEnvironment=["auto","cpu","cpu","cpu"]);

%% 3. Training options
numEpochs = 3;
learnRate = 0.01;
validationFrequency = 50;

trailingAvg = [];
trailingAvgSq = [];

monitor = trainingProgressMonitor( ...
    Metrics=["TrainingLoss","ValidationLoss"], ...
    Info="Epoch", ...
    XLabel="Iteration");
groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"]);

%% 4. Custom training loop
iteration = 0;
for epoch = 1:numEpochs
    shuffle(mbq);

    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        [XTrainBatch,ATrainBatch,numNodesBatch,TTrainBatch] = next(mbq);

        [loss,gradients] = dlfeval(@modelLossGraphSAGE,parameters,XTrainBatch,ATrainBatch,numNodesBatch,TTrainBatch);

        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg,trailingAvgSq,iteration,learnRate);

        recordMetrics(monitor,iteration,TrainingLoss=double(loss));
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100*epoch/numEpochs;

        if mod(iteration,validationFrequency) == 0
            [XVal,AVal,numNodesVal] = preprocessGraphMiniBatch(jetNodeFeatures(idxVal),jetAdjacency(idxVal));
            XVal = dlarray(XVal);
            YVal = modelGraphSAGE(parameters,XVal,AVal,numNodesVal);
            TVal = labels(idxVal);
            lossVal = crossentropy(YVal,TVal,ClassificationMode="multilabel",DataFormat="BC");
            recordMetrics(monitor,iteration,ValidationLoss=double(lossVal));
        end
    end
end

if ~isfolder('../models'); mkdir('../models'); end
save('../models/graphsage_model.mat',"parameters");

fprintf("\nDone. Model saved to models/graphsage_model.mat\n");
fprintf("NEXT STEP: run s5_evaluate_baseline.m\n");
