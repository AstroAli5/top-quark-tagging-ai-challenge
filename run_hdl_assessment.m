function report = run_hdl_assessment(outputDir)
%RUN_HDL_ASSESSMENT Estimate the shipped CNN on a processor model, not a board.
% No synthesis, generated HDL, deployment, or measured FPGA throughput is claimed.
    setupProject;
    root = fileparts(mfilename('fullpath'));
    if nargin < 1, outputDir = fullfile(root,'runs','hdl-assessment'); end
    if isfolder(outputDir)
        error('topquark:ExistingRun','Choose a new outputDir.');
    end
    mkdir(outputDir);
    report = struct('matlabVersion',version,'products',ver, ...
        'workflowRun',getenv('GITHUB_RUN_ID'),'status','not_started', ...
        'hardwareExecuted',false,'hdlGenerated',false, ...
        'scope','Processor-model estimate only; no physical FPGA or vendor synthesis tools');
    [~,commit] = system('git rev-parse HEAD'); report.codeCommit = strtrim(commit);
    checkpoint = fullfile(root,'checkpoints','seed-101','cnn_model.mat');
    report.checkpointSHA256 = projectFileSHA256(checkpoint);
    model = load(checkpoint,'netCNN');
    try
        processor = dlhdl.ProcessorConfig;
        report.targetPlatform = processor.TargetPlatform;
        report.targetFrequencyMHz = processor.TargetFrequency;
        report.processorDataType = processor.ProcessorDataType;
        report.estimatorAssumptions = [ ...
            'Toolbox model assumes dedicated programmable-logic DDR access; ', ...
            'does not establish host preprocessing, transfer, contention or end-to-end latency.'];
        performance = processor.estimatePerformance(model.netCNN,FrameCount=1);
        writetable(performance,fullfile(outputDir,'estimated_performance.csv'),WriteRowNames=true);
        save(fullfile(outputDir,'processor_estimate.mat'),'processor','performance');
        report.status = 'estimated';
    catch exception
        report.status = 'blocked';
        report.errorIdentifier = exception.identifier;
        report.errorMessage = exception.message;
        writeProjectJSON(fullfile(outputDir,'report.json'),report);
        rethrow(exception);
    end
    writeProjectJSON(fullfile(outputDir,'report.json'),report);
    disp(report);
end
