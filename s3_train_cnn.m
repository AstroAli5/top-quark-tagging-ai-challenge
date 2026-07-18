% s3_train_cnn.m
% -------------------------------------------------------------------------
% STEP 3 of 7 — Train the CNN baseline on the 2D jet images.
%
% This is the more standard half of the baseline (an ordinary image
% classifier), matching the "ResNet-based CNN on image representations"
% half of the official GraphSAGE Classifier for Top Quark Tagging demo
% this project is inspired by. A compact CNN is used here rather than a
% full ResNet so a first run finishes in a reasonable time on a laptop;
% swap in more/deeper convolution blocks later if you want to push
% accuracy further.
%
% Toolboxes required: Deep Learning Toolbox.
% -------------------------------------------------------------------------

clear; clc;

fprintf("Step 3/7: Training CNN baseline...\n");

load('../data/jet_images.mat',"jetImages","labels","idxTrain","idxVal","idxTest");

XTrain = jetImages(:,:,:,idxTrain);
XVal   = jetImages(:,:,:,idxVal);

YTrain = categorical(labels(idxTrain));
YVal   = categorical(labels(idxVal));

%% Define a compact CNN
layers = [
    imageInputLayer([32 32 1],Normalization="zscore")

    convolution2dLayer(3,16,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(3,64,Padding="same")
    batchNormalizationLayer
    reluLayer
    globalAveragePooling2dLayer

    fullyConnectedLayer(2)
];

options = trainingOptions("adam", ...
    MaxEpochs=15, ...
    MiniBatchSize=128, ...
    InitialLearnRate=1e-3, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal,YVal}, ...
    ValidationFrequency=50, ...
    Plots="training-progress", ...
    Verbose=false);

fprintf("Training...\n");
netCNN = trainnet(XTrain,YTrain,layers,"crossentropy",options);

if ~isfolder('../models'); mkdir('../models'); end
save('../models/cnn_model.mat',"netCNN");

fprintf("Done. Model saved to models/cnn_model.mat\n");
fprintf("NEXT STEP: run s4_train_graphsage.m (or go straight to s5 if you already trained it)\n");
