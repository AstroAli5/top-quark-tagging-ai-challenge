function [network,description] = project238Network(cfg)
%PROJECT238NETWORK Keep one image representation for the two CNN variants.
    architecture = 'compact';
    if isfield(cfg,'architecture'), architecture = cfg.architecture; end
    architecture = validatestring(architecture,{'compact','resnet18'});
    if strcmp(architecture,'resnet18')
        % Use the named MathWorks architecture, initialized from scratch.
        % Adapt only the input/stem, pooling and binary classification head.
        graph = resnet18(Weights='none');
        graph = replaceLayer(graph,'data',imageInputLayer( ...
            [cfg.imageSize cfg.imageSize 1],Normalization='zscore',Name='data'));
        graph = replaceLayer(graph,'conv1',convolution2dLayer(7,64, ...
            Stride=2,Padding=3,NumChannels=1,Name='conv1'));
        graph = replaceLayer(graph,'pool5',globalAveragePooling2dLayer(Name='pool5'));
        graph = replaceLayer(graph,'fc1000',fullyConnectedLayer(2,Name='fc1000'));
        graph = removeLayers(graph,'ClassificationLayer_predictions');
        network = dlnetwork(graph);
        description = 'Named ResNet18, random initialization; grayscale input, global pooling, two classes';
    else
        network = [
            imageInputLayer([cfg.imageSize cfg.imageSize 1],Normalization='zscore')
            convolution2dLayer(3,16,Padding='same')
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,Stride=2)
            convolution2dLayer(3,32,Padding='same')
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,Stride=2)
            convolution2dLayer(3,64,Padding='same')
            batchNormalizationLayer
            reluLayer
            globalAveragePooling2dLayer
            fullyConnectedLayer(2)
            softmaxLayer];
        description = 'Compact three-convolution CNN';
    end
end
