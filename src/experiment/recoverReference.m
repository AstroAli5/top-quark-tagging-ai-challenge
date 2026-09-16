function recoverReference(seed)
%RECOVERREFERENCE Evaluate a fully trained reference using the corrected reader.
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    officialDir = fullfile(root,'data','official');
    output = fullfile(root,'runs',sprintf('reference_%d',seed));
    manifest = jsondecode(fileread(fullfile(officialDir,'manifest.json')));
    assert(strcmp(manifest.fitting_sha256,'b8911d9b5646ff8e031c1baf2946f0c3b532ba382574540d2f37679575f16a6e'));
    model = load(fullfile(output,'models','winner_reference.mat'));
    progress = load(fullfile(output,'models','training_progress.mat'));
    assert(isequal(string(progress.names),"ResNeXt-SE reference") && isscalar(progress.times) && progress.times > 0);
    assert(model.cfg.winnerSeed == seed && model.cfg.winnerEpochs == 12 && model.normalization.trainingJets == 50000);
    cfg = model.cfg;
    cfg.dataDir = fullfile(output,'data'); cfg.modelsDir = fullfile(output,'models');
    cfg.resultsDir = fullfile(output,'results'); cfg.inputFile = fullfile(officialDir,'fitting.mat');
    cfg.experimentModels = {'ResNeXt-SE reference'};
    cfg.recoveredFromWorkflow = '35094350524';
    cfg.trainingCommit = '0470bd1368f22c976096ad74f8c5c3538ab86195';
    evaluateOfficialTest(cfg,officialDir,manifest,seed,progress.times);
    path = fullfile(cfg.resultsDir,'metadata.json');
    metadata = jsondecode(fileread(path));
    metadata.trainingProvenance.timingSource = 'training_progress.mat: elapsed training-function wall time';
    fid = fopen(path,'w'); assert(fid>=0); cleanup = onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(metadata,PrettyPrint=true));
end
