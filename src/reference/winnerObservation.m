function observation = winnerObservation(fourVectors,normalization,cfg,label)
%WINNEROBSERVATION One normalized image and row of four global features.
    [image,features] = buildWinnerJetFeatures(fourVectors,cfg);
    image = sign(image).*log1p(abs(image));
    image = (image-normalization.imageMean)./normalization.imageStd;
    features = (features-normalization.featureMean)./normalization.featureStd;
    observation = {image,features};
    if nargin > 3
        observation{3} = categorical(label,[0 1],{'0','1'});
    end
end
