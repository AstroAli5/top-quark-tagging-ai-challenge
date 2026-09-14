function run_all(cfg)
%RUN_ALL Execute all seven stages. Existing outputs in cfg are overwritten.
%   Run from the project folder, or add that folder to the MATLAB path.
%   Input data must already exist; see scripts/convert_dataset.py.
    if nargin < 1, cfg = projectConfig; end
    if verLessThan('matlab','24.1') || isempty(ver('nnet'))
        error('topquark:Requirements', ...
            'MATLAB R2024a or later and Deep Learning Toolbox are required.');
    end
    startedAt = char(datetime('now','TimeZone','UTC'));
    stages = {@s1_prepare_data,@s2_build_representations,@s3_train_cnn, ...
        @s4_train_graphsage,@s5_evaluate_baseline,@s6_robustness_test,@s7_explainability};
    for step = 1:numel(stages)
        stages{step}(cfg);
    end
    split = load(fullfile(cfg.dataDir,'jet_split.mat'), ...
        'idxTrain','idxVal','idxTest','datasetId','sourceInfo');
    metadata = struct('startedAtUTC',startedAt, ...
        'finishedAtUTC',char(datetime('now','TimeZone','UTC')), ...
        'matlabVersion',version,'configuration',cfg, ...
        'datasetId',split.datasetId,'sourceInfo',split.sourceInfo, ...
        'trainJets',numel(split.idxTrain),'validationJets',numel(split.idxVal), ...
        'testJets',numel(split.idxTest));
    save(fullfile(cfg.resultsDir,'run_metadata.mat'),'metadata');
    fid = fopen(fullfile(cfg.resultsDir,'run_metadata.json'),'w');
    if fid < 0, error('topquark:WriteFailed','Cannot write run metadata.'); end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(metadata,PrettyPrint=true));
    fprintf('Done. Results saved to %s\n',cfg.resultsDir);
end
