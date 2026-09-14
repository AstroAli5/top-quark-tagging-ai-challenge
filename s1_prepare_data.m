function s1_prepare_data(cfg)
%S1_PREPARE_DATA Validate the converted N-by-800 particle table and labels.
    if nargin < 1, cfg = projectConfig; end
    fprintf('Step 1/7: Loading %s\n',cfg.inputFile);
    if ~isfile(cfg.inputFile)
        error('topquark:MissingDataset', ...
            'Dataset not found: %s. Run scripts/convert_dataset.py first (see README).',cfg.inputFile);
    end
    raw = load(cfg.inputFile);
    if ~isfield(raw,'particleData') || ~isfield(raw,'labels')
        error('topquark:InvalidDataset','MAT file requires particleData and labels.');
    end
    validateattributes(raw.particleData,{'numeric'},{'2d','real','finite','nonempty','ncols',800});
    validateattributes(raw.labels,{'numeric','logical'},{'real','finite','vector'});
    particleData = double(raw.particleData);
    labels = double(raw.labels(:));
    numJets = size(particleData,1);
    if numel(labels) ~= numJets || any(labels ~= 0 & labels ~= 1) || numel(unique(labels)) ~= 2
        error('topquark:InvalidDataset','Supply one 0/1 label per jet and include both classes.');
    end
    jetFourVectors = cell(numJets,1);
    removedParticles = 0;
    for j = 1:numJets
        jet = reshape(particleData(j,:),4,200).';
        if any(jet(:,1) < 0)
            error('topquark:InvalidDataset','Negative particle energy in jet %d.',j);
        end
        nonpadding = any(jet ~= 0,2);
        keep = jet(:,1) > 0 & hypot(jet(:,2),jet(:,3)) > 0;
        removedParticles = removedParticles + sum(nonpadding & ~keep);
        jetFourVectors{j} = jet(keep,:);
    end
    % Identify the prepared dataset so stale models cannot be mixed with it.
    datasetId = char(datetime('now','TimeZone','UTC', ...
        'Format','yyyyMMdd''T''HHmmss.SSSSSSSSS'));
    sourceInfo = struct('inputFile',cfg.inputFile,'inputJets',numJets, ...
        'removedNonpaddingParticles',removedParticles);
    if isfield(raw,'provenance_json')
        sourceInfo.conversion = jsondecode(char(raw.provenance_json));
    else
        sourceInfo.conversion = struct('status','Source provenance not supplied.');
    end
    if ~isfolder(cfg.dataDir), mkdir(cfg.dataDir); end
    save(fullfile(cfg.dataDir,'jets_raw.mat'), ...
        'jetFourVectors','labels','datasetId','sourceInfo','-v7.3');
    fprintf('Loaded %d jets; signal %d, background %d. Removed %d invalid/zero-pT constituents.\n', ...
        numJets,sum(labels==1),sum(labels==0),removedParticles);
end
