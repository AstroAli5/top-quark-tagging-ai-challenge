function normalization = fitWinnerNormalization(trainingJets,cfg)
%FITWINNERNORMALIZATION Streaming statistics using training jets only.
% Signed log compression limits the range of the 12 image channels.
    assert(~isempty(trainingJets),'Training jets must not be empty.');
    imageSum = zeros(1,12); imageSquares = zeros(1,12);
    featureSum = zeros(1,4); featureSquares = zeros(1,4);
    pixels = 0;
    for j = 1:numel(trainingJets)
        [image,features] = buildWinnerJetFeatures(trainingJets{j},cfg);
        values = reshape(double(image),[],12);
        values = sign(values).*log1p(abs(values));
        imageSum = imageSum + sum(values,1);
        imageSquares = imageSquares + sum(values.^2,1);
        featureSum = featureSum + double(features);
        featureSquares = featureSquares + double(features).^2;
        pixels = pixels+size(values,1);
    end
    normalization.imageMean = single(reshape(imageSum/pixels,1,1,12));
    normalization.imageStd = single(reshape(sqrt(max( ...
        imageSquares/pixels-(imageSum/pixels).^2,1e-8)),1,1,12));
    normalization.featureMean = single(featureSum/numel(trainingJets));
    normalization.featureStd = single(sqrt(max(featureSquares/numel(trainingJets) ...
        -(featureSum/numel(trainingJets)).^2,1e-8)));
    normalization.trainingJets = numel(trainingJets);
    normalization.version = 'aligned-12ch-radial-v1';
end
