function [jets,labels,rows] = readJetChunk(path)
%READJETCHUNK Decode one official test chunk without retaining the full dataset.
    raw = load(path);
    labels = double(raw.labels(:)); rows = double(raw.sourceRows(:));
    validateattributes(raw.particleData,{'numeric'},{'2d','finite','ncols',800});
    assert(size(raw.particleData,1) == numel(labels) && numel(rows) == numel(labels));
    assert(all(ismember(labels,[0 1])),'Invalid test labels.');
    jets = cell(numel(labels),1);
    for j = 1:numel(jets)
        fv = double(reshape(raw.particleData(j,:),4,200).');
        assert(all(fv(:,1) >= 0),'Negative test energy.');
        fv = fv(fv(:,1)>0 & hypot(fv(:,2),fv(:,3))>0,:);
        % Nonempty one/two-particle jets are valid inputs. The graph builder
        % already caps neighbors at n-1, including an isolated single node.
        if isempty(fv)
            error('topquark:InvalidTestJet','Official test row %d has no usable particles.',rows(j));
        end
        jets{j} = fv;
    end
end
