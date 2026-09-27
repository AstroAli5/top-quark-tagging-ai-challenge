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
    python = pyenv;
    if python.Status == "Loaded" && python.ExecutionMode ~= "OutOfProcess"
        error('topquark:PythonMode', ...
            'Restart MATLAB and select pyenv(ExecutionMode="OutOfProcess") before using Python.');
    end
    if python.Status ~= "Loaded"
        pyenv(ExecutionMode="OutOfProcess"); % isolate MATLAB/PyTables HDF5 libraries
    end
    timer = tic;
    fprintf('Preparing separate official partitions as bounded Parquet chunks.\n');
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
    progress = struct('completedStage','parquet','parquetPreparationSeconds',prepareSeconds);
    writeProjectJSON(fullfile(cfg.dataDir,'stage_times.json'),progress);
    fprintf('Parquet preparation: %.1f seconds. Creating labelled images with tall.\n',prepareSeconds);
    timer = tic;
    manifest = parquetJetsToImages(cfg);
    imageSeconds = toc(timer);
    progress.completedStage = 'images'; progress.imagePreparationSeconds = imageSeconds;
    writeProjectJSON(fullfile(cfg.dataDir,'stage_times.json'),progress);
    fprintf('Image preparation: %.1f seconds. Training from imageDatastore.\n',imageSeconds);
    report = trainProject238(cfg,manifest);
    report.parquetPreparationSeconds = prepareSeconds;
    report.imagePreparationSeconds = imageSeconds;
    report.peakResidentKiB = [];
    report.resourceMeasurement = 'Linux MATLAB process VmHWM; excludes separate Python process';
    report.pythonPeakResidentKiB = manifest.python_peak_resident_kib;
    if isfile('/proc/self/status')
        match = regexp(fileread('/proc/self/status'),'VmHWM:\s+(\d+)\s+kB','tokens','once');
        if ~isempty(match), report.peakResidentKiB = str2double(match{1}); end
    end
    writeProjectJSON(fullfile(cfg.outputDir,'report.json'),report);
end
