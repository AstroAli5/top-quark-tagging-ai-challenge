function report = explainSavedModels(runDirs,officialDir,outputDir,options)
%EXPLAINSAVEDMODELS Explain frozen CNN/GraphSAGE checkpoints without training.
%   setupProject; explainSavedModels({'runs/core_101','runs/core_202', ...
%       'runs/core_303'},'data/official','results/explanations')
%   Uses the first 10,000 official test jets, three feature permutations,
%   and five radial masks. Variation is summarized across training seeds.
    if nargin < 4, options = struct; end
    defaults = struct('maxJets',10000,'permutationSeeds',[11 21 31], ...
        'radiusFractions',[1 0.75 0.5 0.35 0.2]);
    for field = reshape(fieldnames(options),1,[])
        assert(isfield(defaults,field{1}),'Unknown explanation option.');
        defaults.(field{1}) = options.(field{1});
    end
    options = defaults;
    validateattributes(options.maxJets,{'numeric'},{'scalar','integer','positive'});
    validateattributes(options.permutationSeeds,{'numeric'},{'vector','integer','nonnegative','nonempty'});
    validateattributes(options.radiusFractions,{'numeric'},{'vector','positive','<=',1,'nonempty'});
    assert(numel(unique(options.permutationSeeds)) == numel(options.permutationSeeds));
    assert(options.radiusFractions(1) == 1 && all(diff(options.radiusFractions)<0), ...
        'Radii must start at 1 and decrease without duplicates.');
    if isfolder(outputDir) && ~isempty(dir(fullfile(outputDir,'*.*')))
        error('topquark:ExistingExplanation','Choose a new explanation output folder.');
    end
    runDirs = cellstr(string(runDirs));
    assert(~isempty(runDirs),'Provide at least one completed core run.');
    manifest = jsondecode(fileread(fullfile(officialDir,'manifest.json')));
    assert(strcmp(manifest.dataset,'10.5281/zenodo.2603256'),'Unexpected dataset.');
    n = min(options.maxJets,manifest.test_count);
    [jets,labels,rows,blocks] = readSample(officialDir,manifest,n);
    assert(numel(unique(labels)) == 2,'The explanation sample needs both classes.');
    featureNames = ["deltaEta";"deltaPhi";"log(pT)";"log(E)"];
    permutationSeeds = options.permutationSeeds;
    radiusFractions = options.radiusFractions;
    seedCount = numel(runDirs);
    seeds = zeros(seedCount,1);
    baselineAUC = zeros(seedCount,2);
    dropBySeed = zeros(seedCount,4);
    radialBySeed = zeros(seedCount,numel(radiusFractions));
    permutationTable = table; radiusTable = table;
    provenance = cell(seedCount,1);
    % CPU kernels can differ by a few float32 rounding units across runners.
    % Also require unchanged decisions and near-identical ranking metrics.
    probabilityTolerance = 64*double(eps('single'));
    aucTolerance = 1e-6;
    if ~isfolder(outputDir), mkdir(outputDir); end
    for s = 1:seedCount
        runDir = runDirs{s};
        metadata = jsondecode(fileread(fullfile(runDir,'results','metadata.json')));
        verifySourceManifest(metadata.manifest,manifest);
        cnnPath = fullfile(runDir,'models','cnn_model.mat');
        sagePath = fullfile(runDir,'models','graphsage_model.mat');
        cnn = load(cnnPath); sage = load(sagePath);
        verifyDatasetIds(cnn,sage,metadata);
        seed = metadata.trainingSeed;
        assert(cnn.cfg.cnnSeed == seed && sage.cfg.graphSeed == seed,'Training seed mismatch.');
        assert(~ismember(seed,seeds(1:s-1)),'Duplicate training seed.');
        seeds(s) = seed;
        cfg = cnn.cfg;
        cfg.graphBatchSize = sage.cfg.graphBatchSize;
        saved = load(fullfile(runDir,'results','clean_predictions.mat'));
        assert(saved.seed == seed && numel(saved.rows) == manifest.test_count, ...
            'Expected predictions from the completed official evaluation.');
        if ~isequal(rows,double(saved.rows(1:n))) || ~isequal(labels,double(saved.labels(1:n)))
            error('topquark:ExplanationMismatch','Test rows or labels differ from the saved evaluation.');
        end
        cnnColumn = find(string(saved.names) == "CNN");
        graphColumn = find(string(saved.names) == "GraphSAGE");
        assert(isscalar(cnnColumn) && isscalar(graphColumn),'Missing core prediction columns.');
        images = zeros(cfg.imageSize,cfg.imageSize,1,n,'single');
        features = cell(n,1); adjacency = cell(n,1);
        for j = 1:n
            images(:,:,:,j) = single(buildJetImage(jets{j},cfg.imageSize));
            [f,a] = buildJetGraph(jets{j},sage.cfg.kNeighbors);
            features{j} = single(f); adjacency{j} = sparse(a);
        end
        cleanProbabilities = [cnnScores(cnn,images,cfg,blocks), ...
            graphScores(sage,features,adjacency,cfg,blocks)];
        originalProbabilities = saved.probabilities(1:n,[cnnColumn graphColumn]);
        difference = max(abs(cleanProbabilities-originalProbabilities),[],'all');
        if difference > probabilityTolerance || ~isequal(cleanProbabilities>=0.5,originalProbabilities>=0.5)
            error('topquark:ExplanationMismatch','Clean predictions changed by %.3g; aborting explanations.',difference);
        end
        aucDifference = zeros(1,2);
        for m = 1:2
            [~,~,baselineAUC(s,m)] = computeROC(cleanProbabilities(:,m),labels);
            [~,~,originalAUC] = computeROC(originalProbabilities(:,m),labels);
            aucDifference(m) = abs(baselineAUC(s,m)-originalAUC);
            if aucDifference(m) > aucTolerance
                error('topquark:ExplanationMismatch','Clean AUC changed beyond the numerical tolerance.');
            end
        end
        nodeCounts = cellfun(@(x) size(x,1),features);
        allFeatures = vertcat(features{:});
        permutationProbabilities = zeros(n,4,numel(permutationSeeds));
        drops = zeros(numel(permutationSeeds),4);
        for r = 1:numel(permutationSeeds)
            stream = RandStream('mt19937ar','Seed',permutationSeeds(r));
            for f = 1:4
                shuffled = allFeatures;
                shuffled(:,f) = shuffled(randperm(stream,size(shuffled,1)),f);
                perturbed = mat2cell(shuffled,nodeCounts,size(shuffled,2));
                p = graphScores(sage,perturbed,adjacency,cfg,blocks);
                permutationProbabilities(:,f,r) = p;
                [~,~,auc] = computeROC(p,labels);
                drops(r,f) = baselineAUC(s,2)-auc;
                permutationTable = [permutationTable;table(seed,permutationSeeds(r),featureNames(f), ...
                    n,baselineAUC(s,2),auc,drops(r,f),VariableNames= ...
                    {'Seed','PermutationSeed','Feature','TestJets','BaselineAUC','PerturbedAUC','AUCDrop'})]; %#ok<AGROW>
            end
        end
        dropBySeed(s,:) = mean(drops,1);
        [x,y] = meshgrid(1:cfg.imageSize,1:cfg.imageSize);
        radiusMap = hypot(x-(cfg.imageSize+1)/2,y-(cfg.imageSize+1)/2);
        occlusionProbabilities = zeros(n,numel(radiusFractions));
        for r = 1:numel(radiusFractions)
            mask = radiusMap <= radiusFractions(r)*max(radiusMap(:));
            p = cnnScores(cnn,images.*single(mask),cfg,blocks);
            occlusionProbabilities(:,r) = p;
            [~,~,auc] = computeROC(p,labels);
            radialBySeed(s,r) = auc;
            radiusTable = [radiusTable;table(seed,radiusFractions(r),n,baselineAUC(s,1),auc, ...
                baselineAUC(s,1)-auc,VariableNames= ...
                {'Seed','RadiusFractionKept','TestJets','BaselineAUC','AUC','AUCDrop'})]; %#ok<AGROW>
        end
        assert(isequal(occlusionProbabilities(:,1),cleanProbabilities(:,1)), ...
            'The full-radius mask must preserve every clean prediction.');
        save(fullfile(outputDir,sprintf('predictions_seed_%d.mat',seed)), ...
            'seed','rows','labels','featureNames','permutationSeeds','radiusFractions', ...
            'cleanProbabilities','permutationProbabilities','occlusionProbabilities','-v7');
        provenance{s} = struct('trainingSeed',seed,'datasetId',cnn.datasetId, ...
            'cnnCheckpointSHA256',fileSHA256(cnnPath),'graphCheckpointSHA256',fileSHA256(sagePath), ...
            'sourceEvaluation',metadata,'maxCleanProbabilityDifference',difference, ...
            'cleanAUCDifference',aucDifference,'changedCleanDecisions',0);
        fprintf('Explained seed %d: %d test jets, %d permutations, %d radii; clean delta %.3g.\n', ...
            seed,n,numel(permutationSeeds),numel(radiusFractions),difference);
    end
    featureSummary = table(featureNames,mean(dropBySeed,1).',std(dropBySeed,0,1).', ...
        repmat(seedCount,4,1),repmat(numel(permutationSeeds),4,1),VariableNames= ...
        {'Feature','MeanAUCDrop','SeedSD','TrainingSeeds','PermutationRepeats'});
    radiusSummary = table(radiusFractions(:),mean(radialBySeed,1).',std(radialBySeed,0,1).', ...
        repmat(seedCount,numel(radiusFractions),1),VariableNames= ...
        {'RadiusFractionKept','MeanAUC','SeedSD','TrainingSeeds'});
    writetable(permutationTable,fullfile(outputDir,'permutation_per_repeat.csv'));
    writetable(radiusTable,fullfile(outputDir,'occlusion_per_seed.csv'));
    writetable(featureSummary,fullfile(outputDir,'feature_summary.csv'));
    writetable(radiusSummary,fullfile(outputDir,'occlusion_summary.csv'));
    description = sprintf('%s training jets | %s official test jets | %d training seeds', ...
        string(manifest.train_count),string(n),seedCount);
    fig = figure('Visible','off','Color','w','Position',[100 100 900 560]);
    cleanup = onCleanup(@() close(fig));
    bar(featureSummary.MeanAUCDrop,'FaceColor',[0.85 0.33 0.10]); hold on;
    errorbar(1:4,featureSummary.MeanAUCDrop,featureSummary.SeedSD,'k.','LineWidth',1.5);
    xticks(1:4); xticklabels(featureNames); grid on;
    ylabel('AUC drop after shuffling (mean +/- training-seed SD)');
    title('GraphSAGE feature permutation: fixed graph edges'); subtitle(description);
    exportgraphics(fig,fullfile(outputDir,'graphsage_feature_importance.png'),'Resolution',180);
    clear cleanup;
    fig = figure('Visible','off','Color','w','Position',[100 100 900 560]);
    cleanup = onCleanup(@() close(fig));
    errorbar(radiusFractions,radiusSummary.MeanAUC,radiusSummary.SeedSD,'-o','LineWidth',1.8);
    set(gca,'XDir','reverse'); xticks(sort(radiusFractions)); grid on;
    xlabel('Fraction of center-to-corner image radius retained');
    ylabel('AUC (mean +/- training-seed SD)');
    title('CNN radial occlusion: frozen final checkpoints'); subtitle(description);
    exportgraphics(fig,fullfile(outputDir,'cnn_radial_occlusion.png'),'Resolution',180);
    report = struct('analysis','Post-hoc sensitivity; no retraining or model selection', ...
        'options',options,'testJets',n,'firstSourceRow',rows(1),'lastSourceRow',rows(end), ...
        'cleanProbabilityTolerance',probabilityTolerance,'cleanAUCTolerance',aucTolerance, ...
        'trainingSeeds',seeds,'trainingJets',manifest.train_count,'validationJets',manifest.val_count, ...
        'permutationAggregation','Average repeats within each training seed, then mean and sample SD across training seeds', ...
        'graphEdges','Unchanged; one feature column shuffled across all constituent nodes', ...
        'occlusion','Pixel values outside each central disk set to zero; training normalization unchanged', ...
        'matlabVersion',version,'codeCommit',getenv('GITHUB_SHA'),'workflowRun',getenv('GITHUB_RUN_ID'), ...
        'createdAtUTC',char(datetime('now','TimeZone','UTC')),'provenance',{provenance});
    fid = fopen(fullfile(outputDir,'metadata.json'),'w'); assert(fid>=0);
    fileCleanup = onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
    disp(featureSummary); disp(radiusSummary);
end

function [jets,labels,rows,blocks] = readSample(officialDir,manifest,n)
    jets = cell(n,1); labels = zeros(n,1); rows = zeros(n,1); position = 0;
    blocks = {};
    for c = 1:numel(manifest.test_chunks)
        chunk = manifest.test_chunks(c);
        path = fullfile(officialDir,chunk.file);
        if isfield(chunk,'sha256')
            assert(strcmp(fileSHA256(path),chunk.sha256),'Prepared test chunk checksum mismatch.');
        end
        [batch,y,ids] = readJetChunk(path);
        assert(chunk.start == position && isequal(ids,(chunk.start:chunk.stop-1).'), ...
            'Test rows have gaps, duplicates, or changed order.');
        count = min(numel(batch),n-position);
        idx = position+(1:count);
        blocks{end+1} = idx; %#ok<AGROW>
        jets(idx) = batch(1:count); labels(idx) = y(1:count); rows(idx) = ids(1:count);
        position = position+count;
        if position == n, break; end
    end
    assert(position == n,'Incomplete explanation sample.');
end

function p = cnnScores(cnn,images,cfg,blocks)
    % Preserve the original test-chunk batch boundaries and float arithmetic.
    p = zeros(size(images,4),1);
    for b = 1:numel(blocks)
        idx = blocks{b};
        p(idx) = predictCNN(cnn.netCNN,images(:,:,:,idx),cnn.classNames,cfg.cnnBatchSize);
    end
end

function p = graphScores(sage,features,adjacency,cfg,blocks)
    p = zeros(numel(features),1);
    for b = 1:numel(blocks)
        idx = blocks{b};
        p(idx) = predictGraphSAGE(sage.parameters,features(idx),adjacency(idx),cfg.graphBatchSize);
    end
end

function verifySourceManifest(saved,current)
    % Regenerating a MAT file changes its header timestamp and byte hash.
    % Require identical source partitions/row intervals, check the current
    % chunk hashes on read, and compare predictions with the frozen record.
    for field = {'dataset','train_count','val_count','test_count','full_official_test'}
        assert(isequaln(saved.(field{1}),current.(field{1})),'Checkpoint data manifest mismatch.');
    end
    if isfield(saved,'sources')
        assert(isfield(current,'sources') && isequaln(saved.sources,current.sources), ...
            'Source partition hashes changed.');
    end
    a = saved.test_chunks; b = current.test_chunks;
    if isfield(a,'sha256'), a = rmfield(a,'sha256'); end
    if isfield(b,'sha256'), b = rmfield(b,'sha256'); end
    assert(isequaln(a,b),'Official test chunk boundaries changed.');
end

function value = fileSHA256(path)
    fid = fopen(path,'rb'); assert(fid>=0,'Cannot open input file.');
    cleanup = onCleanup(@() fclose(fid));
    digest = java.security.MessageDigest.getInstance('SHA-256');
    while ~feof(fid)
        digest.update(fread(fid,1024*1024,'*int8'));
    end
    value = lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2).',1,[]));
end
