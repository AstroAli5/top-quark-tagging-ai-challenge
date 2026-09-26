function report = run_project238(cfg)
%RUN_PROJECT238 MATLAB-hosted HDF5 -> Parquet -> tall images -> CNN.
% Inf selects the full source partition. Existing fitted outputs are protected.
    setupProject;
    if nargin < 1, cfg = project238Config; end
    if isMATLABReleaseOlderThan('R2024a') || isempty(ver('nnet'))
        error('topquark:Requirements','MATLAB R2024a+ and Deep Learning Toolbox required.');
    end
    if ~usejava('jvm')
        error('topquark:JVMRequired','Start MATLAB with its JVM enabled for provenance hashing.');
    end
    if isfolder(cfg.outputDir)
        error('topquark:ExistingRun','Choose a new outputDir to preserve existing results.');
    end
    timer = tic;
    counts = struct('train',cfg.trainCount,'val',cfg.validationCount,'test',cfg.testCount);
    % jsonencode maps Inf to null; Python interprets null as all source rows.
    scripts = fullfile(fileparts(mfilename('fullpath')),'scripts');
    parquetDir = fullfile(cfg.dataDir,'parquet');
    pyrun(["import sys, pandas as pd", ...
        "sys.path.insert(0, str(scripts)) if str(scripts) not in sys.path else None", ...
        "from prepare_parquet import prepare", ...
        "manifest = prepare(str(raw), str(output), str(counts), int(chunk_rows))"], ...
        scripts=string(scripts),raw=string(cfg.rawDir),output=string(parquetDir), ...
        counts=string(jsonencode(counts)),chunk_rows=cfg.chunkRows);
    prepareSeconds = toc(timer);
    timer = tic;
    manifest = parquetJetsToImages(cfg);
    imageSeconds = toc(timer);
    report = trainProject238(cfg,manifest);
    report.parquetPreparationSeconds = prepareSeconds;
    report.imagePreparationSeconds = imageSeconds;
    report.peakResidentKiB = [];
    report.resourceMeasurement = 'Linux process VmHWM, including preparation and training';
    if isfile('/proc/self/status')
        match = regexp(fileread('/proc/self/status'),'VmHWM:\s+(\d+)\s+kB','tokens','once');
        if ~isempty(match), report.peakResidentKiB = str2double(match{1}); end
    end
    writeProjectJSON(fullfile(cfg.outputDir,'report.json'),report);
end
