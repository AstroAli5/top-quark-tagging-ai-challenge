function ds = winnerDatastore(jets,labels,normalization,cfg)
%WINNERDATASTORE Generate images on demand without a full 12-channel cache.
    assert(numel(jets) == numel(labels),'Each jet needs one label.');
    indices = arrayDatastore((1:numel(jets)).',OutputType='same',ReadSize=1);
    ds = transform(indices,@(j) winnerObservation(jets{j},normalization,cfg,labels(j)));
end
