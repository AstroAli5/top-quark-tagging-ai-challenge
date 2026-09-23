function runScaledExperiment(seed,officialDir,outputRoot)
%RUNSCALEDEXPERIMENT Core-only study with Linux process peak-memory evidence.
%   Data size is taken from the prepared manifest. Training rules are those
%   of run_experiment; no GPU or full-data scalability claim is implied.
    timer = tic;
    run_experiment(seed,officialDir,outputRoot, ...
        struct('experimentModels',{{'CNN','GraphSAGE'}}));
    resource = struct('wallSeconds',toc(timer),'platform',computer, ...
        'matlabVersion',version,'peakResidentKiB',[], ...
        'measurement','Linux /proc/self/status VmHWM for the MATLAB process');
    if isfile('/proc/self/status')
        match = regexp(fileread('/proc/self/status'),'VmHWM:\s+(\d+)\s+kB','tokens','once');
        if ~isempty(match), resource.peakResidentKiB = str2double(match{1}); end
    end
    if isempty(resource.peakResidentKiB)
        resource.measurement = 'Peak resident memory unavailable on this platform';
    end
    file = fullfile(outputRoot,'results','resources.json');
    fid = fopen(file,'w'); assert(fid>=0,'Cannot write resource measurement.');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(resource,PrettyPrint=true));
end
