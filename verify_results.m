function result = verify_results(scope)
%VERIFY_RESULTS Restore seed-101 models on 256 official test jets, without fitting.
    setupProject;
    requireToolboxes("restore the saved models");
    if nargin < 1, scope = "cnn"; end
    scope = validatestring(scope,{'cnn','all'});
    folder = fullfile(fileparts(mfilename('fullpath')),'checkpoints');
    manifest = jsondecode(fileread(fullfile(folder,'manifest.json')));
    for j = 1:numel(manifest.files)
        if strcmp(scope,'cnn') && ~ismember(string(manifest.files(j).file), ...
                ["seed-101/cnn_model.mat","quick_test.mat"]), continue; end
        item = manifest.files(j);
        assert(strcmp(projectFileSHA256(fullfile(folder,item.file)),item.sha256), ...
            'topquark:CheckpointHash','Checkpoint or sample checksum mismatch.');
    end
    models.cnn = load(fullfile(folder,'seed-101','cnn_model.mat'));
    if strcmp(scope,'all')
        models.sage = load(fullfile(folder,'seed-101','graphsage_model.mat'));
        models.reference = load(fullfile(folder,'seed-101','winner_reference.mat'));
    end
    cfg = models.cnn.cfg; cfg.executionEnvironment = 'cpu';
    cfg.experimentModels = {'CNN'};
    if strcmp(scope,'all'), cfg.experimentModels = {'CNN','GraphSAGE','ResNeXt-SE reference'}; end
    modelCount = numel(cfg.experimentModels);
    sample = fullfile(folder,'quick_test.mat');
    [jets,labels,rows] = readJetChunk(sample);
    expected = load(sample,'expectedProbabilities');
    probabilities = predictThreeModels(models,jets,cfg);
    assert(isequal(rows,(0:255).'),'topquark:QuickSample','Unexpected sample rows.');
    original = expected.expectedProbabilities(:,1:modelCount);
    differenceByModel = max(abs(probabilities-original),[],1);
    difference = max(differenceByModel);
    % Same single-precision budget as the previously audited explanation study.
    % Numerical tolerance never permits a changed class decision or material AUC.
    probabilityTolerance = 64*double(eps('single'));
    accuracy = zeros(modelCount,1); auc = accuracy;
    for j = 1:modelCount
        accuracy(j) = mean((probabilities(:,j)>=.5)==labels);
        [~,~,auc(j)] = computeROC(probabilities(:,j),labels);
        [~,~,originalAUC] = computeROC(original(:,j),labels);
        changed = sum((probabilities(:,j)>=.5) ~= (original(:,j)>=.5));
        fprintf('%s: maximum score difference %.9g, changed decisions %d, AUC difference %.9g.\n', ...
            cfg.experimentModels{j},differenceByModel(j),changed,abs(auc(j)-originalAUC));
        assert(differenceByModel(j)<=probabilityTolerance && changed==0 && abs(auc(j)-originalAUC)<=1e-6, ...
            'topquark:RestorationMismatch','%s predictions changed beyond the recorded numerical budget.', ...
            cfg.experimentModels{j});
    end
    result = table(string(cfg.experimentModels(:)),accuracy,auc, ...
        VariableNames={'Model','Accuracy','AUC'});
    fprintf('Seed 101, trained on 50,000 jets; first 256 official test jets only.\n');
    fprintf('Maximum probability difference from saved results: %.3g\n',difference);
    fprintf('Sample scores are not the 404,000-jet or three-seed mean scores.\n');
    disp(result);
end
