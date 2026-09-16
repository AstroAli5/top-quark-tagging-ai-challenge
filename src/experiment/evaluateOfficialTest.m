function evaluateOfficialTest(cfg,officialDir,manifest,seed,trainingSeconds)
%EVALUATEOFFICIALTEST Freeze models before reading the separate official test set.
    allNames = ["CNN";"GraphSAGE";"ResNeXt-SE reference"];
    fields = {'cnn','sage','reference'};
    files = {'cnn_model.mat','graphsage_model.mat','winner_reference.mat'};
    if isfield(cfg,'experimentModels'), names = string(cfg.experimentModels(:)); else, names = allNames; end
    assert(~isempty(names) && isequal(names,allNames(ismember(allNames,names))), 'Invalid model selection.');
    models = struct;
    for m = 1:3
        if ismember(allNames(m),names), models.(fields{m}) = load(fullfile(cfg.modelsDir,files{m})); end
    end
    loaded = struct2cell(models); verifyDatasetIds(loaded{:});
    probabilities = zeros(manifest.test_count,numel(names));
    labels = zeros(manifest.test_count,1); rows = labels;
    noiseCount = min(cfg.noiseTestJets,manifest.test_count);
    noiseJets = cell(noiseCount,1);
    timer = tic;
    position = 0;
    for c = 1:numel(manifest.test_chunks)
        chunk = manifest.test_chunks(c);
        [jets,y,ids] = readJetChunk(fullfile(officialDir,chunk.file));
        expected = (chunk.start:chunk.stop-1).';
        if ~isequal(ids,expected) || chunk.start ~= position
            error('topquark:InvalidPartition','Official test rows have gaps, duplicates, or changed order.');
        end
        idx = position+(1:numel(jets));
        probabilities(idx,:) = predictThreeModels(models,jets,cfg);
        labels(idx) = y; rows(idx) = ids;
        inNoise = idx <= noiseCount;
        noiseJets(idx(inNoise)) = jets(inNoise);
        position = position+numel(jets);
        if mod(c,10) == 0 || c == numel(manifest.test_chunks)
            fprintf('Official test: %d / %d jets\n',position,manifest.test_count);
        end
    end
    assert(position == manifest.test_count,'Incomplete official test coverage.');
    cleanSeconds = toc(timer);
    metrics = table;
    for m = 1:numel(names)
        [~,~,auc] = computeROC(probabilities(:,m),labels);
        metrics = [metrics;table(seed,names(m),manifest.test_count, ...
            mean((probabilities(:,m)>=0.5)==labels),auc,trainingSeconds(m), ...
            VariableNames={'Seed','Model','TestJets','Accuracy','AUC','TrainingSeconds'})]; %#ok<AGROW>
    end
    if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
    writetable(metrics,fullfile(cfg.resultsDir,'clean_metrics.csv'));
    save(fullfile(cfg.resultsDir,'clean_predictions.mat'),'probabilities','labels','rows','names','seed','-v7');
    disp(metrics);
    noiseLabels = labels(1:noiseCount);
    noiseMetrics = table;
    noisePredictions = zeros(noiseCount,numel(names),numel(cfg.noiseLevels),numel(cfg.noiseSeeds),'single');
    for r = 1:numel(cfg.noiseSeeds)
        for l = 1:numel(cfg.noiseLevels)
            sigma = cfg.noiseLevels(l);
            if sigma == 0
                scores = probabilities(1:noiseCount,:);
            else
                stream = RandStream('mt19937ar','Seed',cfg.noiseSeeds(r));
                scores = zeros(noiseCount,numel(names));
                for first = 1:1000:noiseCount
                    indices = first:min(first+999,noiseCount);
                    jets = cell(numel(indices),1);
                    for j = 1:numel(indices)
                        jets{j} = injectDetectorNoise(noiseJets{indices(j)},sigma,stream);
                    end
                    scores(indices,:) = predictThreeModels(models,jets,cfg);
                end
            end
            noisePredictions(:,:,l,r) = single(scores);
            for m = 1:numel(names)
                [~,~,auc] = computeROC(scores(:,m),noiseLabels);
                noiseMetrics = [noiseMetrics;table(seed,cfg.noiseSeeds(r),sigma,names(m),noiseCount, ...
                    mean((scores(:,m)>=0.5)==noiseLabels),auc, ...
                    VariableNames={'Seed','NoiseSeed','Sigma','Model','TestJets','Accuracy','AUC'})]; %#ok<AGROW>
            end
        end
    end
    noiseLevels = cfg.noiseLevels; noiseSeeds = cfg.noiseSeeds;
    save(fullfile(cfg.resultsDir,'noise_predictions.mat'),'noisePredictions','noiseLabels','noiseLevels','noiseSeeds','seed','-v7');
    writetable(noiseMetrics,fullfile(cfg.resultsDir,'noise_metrics.csv'));
    metadata = struct('trainingSeed',seed,'configuration',cfg,'manifest',manifest, ...
        'matlabVersion',version,'cleanEvaluationSeconds',cleanSeconds, ...
        'createdAtUTC',char(datetime('now','TimeZone','UTC')), ...
        'codeCommit',getenv('GITHUB_SHA'),'workflowRun',getenv('GITHUB_RUN_ID'), ...
        'datasetId',loaded{1}.datasetId,'models',{cellstr(names)});
    if isfield(models,'reference'), metadata.normalization = models.reference.normalization; end
    if isfield(cfg,'recoveredFromWorkflow')
        metadata.trainingProvenance = struct('workflowRun',cfg.recoveredFromWorkflow, ...
            'codeCommit',cfg.trainingCommit,'timingSource','Training-stage start/save log timestamps');
    end
    fid = fopen(fullfile(cfg.resultsDir,'metadata.json'),'w');
    assert(fid>=0,'Cannot write experiment metadata.');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(metadata,PrettyPrint=true));
    fprintf('Noise study: %d seeds x %d strengths x %d jets; metrics saved.\n', ...
        numel(noiseSeeds),numel(noiseLevels),noiseCount);
end
