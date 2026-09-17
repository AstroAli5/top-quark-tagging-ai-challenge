function net = buildWinnerNetwork(cfg)
%BUILDWINNERNETWORK Compact grouped residual CNN with squeeze/excitation.
% Independent architecture, not the winning network's exact weights/layout.
    widths = cfg.winnerWidths;
    validateattributes(widths,{'numeric'},{'vector','integer','positive'});
    validateattributes(cfg.winnerGroups,{'numeric'},{'scalar','integer','positive'});
    assert(all(mod(widths,cfg.winnerGroups) == 0), ...
        'Each winner width must be divisible by the group count.');
    net = dlnetwork;
    net = addLayers(net,[
        imageInputLayer([cfg.winnerImageSize cfg.winnerImageSize 12], ...
            Normalization='none',Name='image')
        convolution2dLayer(3,widths(1),Padding='same',Name='stem_conv')
        batchNormalizationLayer(Name='stem_bn')
        reluLayer(Name='stem_relu')]);
    previous = 'stem_relu';
    for block = 1:numel(widths)
        prefix = sprintf('block%d_',block);
        width = widths(block);
        stride = 1+(block > 1);
        branch = [
            convolution2dLayer(1,width,Name=[prefix 'reduce'])
            batchNormalizationLayer(Name=[prefix 'bn1'])
            reluLayer(Name=[prefix 'relu1'])
            groupedConvolution2dLayer(3,width/cfg.winnerGroups,cfg.winnerGroups, ...
                Padding='same',Stride=stride,Name=[prefix 'group'])
            batchNormalizationLayer(Name=[prefix 'bn2'])
            reluLayer(Name=[prefix 'relu2'])
            convolution2dLayer(1,width,Name=[prefix 'expand'])
            batchNormalizationLayer(Name=[prefix 'bn3'])];
        net = addLayers(net,branch);
        net = connectLayers(net,previous,[prefix 'reduce']);
        attention = [
            globalAveragePooling2dLayer(Name=[prefix 'squeeze'])
            convolution2dLayer(1,max(1,floor(width/8)),Name=[prefix 'se_reduce'])
            reluLayer(Name=[prefix 'se_relu'])
            convolution2dLayer(1,width,Name=[prefix 'se_expand'])
            sigmoidLayer(Name=[prefix 'gate'])];
        net = addLayers(net,attention);
        net = connectLayers(net,[prefix 'bn3'],[prefix 'squeeze']);
        net = addLayers(net,[
            multiplicationLayer(2,Name=[prefix 'scale'])
            additionLayer(2,Name=[prefix 'add'])
            reluLayer(Name=[prefix 'out'])]);
        net = connectLayers(net,[prefix 'bn3'],[prefix 'scale/in1']);
        net = connectLayers(net,[prefix 'gate'],[prefix 'scale/in2']);
        if block == 1
            net = connectLayers(net,previous,[prefix 'add/in2']);
        else
            net = addLayers(net,[
                convolution2dLayer(1,width,Stride=stride,Name=[prefix 'skip'])
                batchNormalizationLayer(Name=[prefix 'skip_bn'])]);
            net = connectLayers(net,previous,[prefix 'skip']);
            net = connectLayers(net,[prefix 'skip_bn'],[prefix 'add/in2']);
        end
        previous = [prefix 'out'];
    end
    net = addLayers(net,[
        globalAveragePooling2dLayer(Name='pool')
        flattenLayer(Name='flatten')
        concatenationLayer(1,2,Name='fusion')
        fullyConnectedLayer(64,Name='head')
        reluLayer(Name='head_relu')
        dropoutLayer(0.1,Name='dropout')
        fullyConnectedLayer(2,Name='logits')
        softmaxLayer(Name='probabilities')]);
    net = connectLayers(net,previous,'pool');
    net = addLayers(net,featureInputLayer(4,Normalization='none',Name='radial'));
    net = connectLayers(net,'radial','fusion/in2');
    net = initialize(net);
    assert(isequal(string(net.InputNames),["image" "radial"]), ...
        'Datastore input order must match the network.');
end
