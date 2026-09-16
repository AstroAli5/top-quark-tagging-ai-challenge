function s2_build_representations(cfg)
%S2_BUILD_REPRESENTATIONS Use identical jets and splits for both models.
    if nargin < 1, cfg = projectConfig; end
    fprintf('Step 2/7: Building images and sparse graphs...\n');
    raw = load(fullfile(cfg.dataDir,'jets_raw.mat'));
    validJets = cellfun(@(x) size(x,1),raw.jetFourVectors) >= 3;
    jetFourVectors = raw.jetFourVectors(validJets);
    labels = raw.labels(validJets);
    sourceRows = find(validJets);
    % Bind checkpoints to the split and representation settings as well as the import.
    datasetId = sprintf('%s-s%d-i%d-k%d',raw.datasetId, ...
        cfg.splitSeed,cfg.imageSize,cfg.kNeighbors);
    sourceInfo = raw.sourceInfo;
    sourceInfo.droppedJets = sum(~validJets);
    numJets = numel(labels);
    if isfield(raw,'partition') && ~isempty(raw.partition)
        partition = raw.partition(validJets);
        idxTrain = find(partition == 1); idxVal = find(partition == 2);
        idxTest = zeros(0,1); % official test data are evaluated separately, in chunks
        sourceInfo.protocol = 'Official train/val files; official test excluded from fitting';
        assert(numel(unique(labels(idxTrain))) == 2 && numel(unique(labels(idxVal))) == 2, ...
            'Both official fitting partitions must contain both classes.');
    else
        [idxTrain,idxVal,idxTest] = stratifiedSplit(labels,cfg.splitSeed);
    end
    jetImages = zeros(cfg.imageSize,cfg.imageSize,1,numJets,'single');
    jetNodeFeatures = cell(numJets,1);
    jetAdjacency = cell(numJets,1);
    for j = 1:numJets
        jetImages(:,:,1,j) = single(buildJetImage(jetFourVectors{j},cfg.imageSize));
        [nf,adj] = buildJetGraph(jetFourVectors{j},cfg.kNeighbors);
        jetNodeFeatures{j} = single(nf);
        jetAdjacency{j} = sparse(adj);
        if mod(j,2000) == 0 || j == numJets
            fprintf('  %d / %d jets\n',j,numJets);
        end
    end
    save(fullfile(cfg.dataDir,'jet_images.mat'),'jetImages','labels', ...
        'idxTrain','idxVal','idxTest','datasetId','cfg','-v7.3');
    save(fullfile(cfg.dataDir,'jet_graphs.mat'),'jetNodeFeatures','jetAdjacency','labels', ...
        'idxTrain','idxVal','idxTest','datasetId','cfg','-v7.3');
    save(fullfile(cfg.dataDir,'jet_split.mat'),'jetFourVectors','labels', ...
        'idxTrain','idxVal','idxTest','sourceRows','sourceInfo','datasetId','cfg','-v7.3');
    fprintf('Split: %d train, %d validation, %d test. Dropped %d near-empty jets.\n', ...
        numel(idxTrain),numel(idxVal),numel(idxTest),sum(~validJets));
end
