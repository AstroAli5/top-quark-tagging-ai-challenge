function reports = run_resnet18_study
%RUN_RESNET18_STUDY Frozen, single-seed comparison with identical jet images.
% This optional pilot does not replace the full-source or repeated-seed studies.
    cfg = project238Config;
    cfg.trainCount = 10000; cfg.validationCount = 2000; cfg.testCount = 10000;
    cfg.epochs = 3; cfg.batchSize = 64; cfg.seed = 101;
    cfg.dataDir = fullfile(cfg.rootDir,'data','resnet18_pilot');
    reports = struct;
    for architecture = ["compact","resnet18"]
        cfg.architecture = char(architecture);
        cfg.outputDir = fullfile(cfg.rootDir,'runs','resnet18-pilot',architecture);
        reports.(architecture) = run_project238(cfg);
    end
end
