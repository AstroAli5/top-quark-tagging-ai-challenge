function result = run_submission(outputDir)
%RUN_SUBMISSION Quick reviewer entry point: saved CNN plus MATLAB study report.
% Uses included files only; no Python, data download or model fitting required.
% Sample predictions and full-study summaries have distinct evaluation scopes.
    root = fileparts(mfilename('fullpath'));
    if nargin < 1, outputDir = fullfile(root,'results','matlab-summary'); end
    timer = tic;
    sample = verify_results;
    [study,paired] = summarize_matlab(fullfile(root,'experiments','official-study'),outputDir);
    result = struct('sample',sample,'study',study,'paired',paired, ...
        'elapsedSeconds',toc(timer),'outputDir',outputDir);
    fprintf('Reviewer check completed in %.2f seconds; no training or download.\n',result.elapsedSeconds);
    fprintf('Restored predictions: seed 101 CNN, 256 test jets.\n');
    fprintf('Study tables: recorded 404,000-jet results across three training seeds.\n');
    fprintf('MATLAB figures and tables: %s\n',outputDir);
end
