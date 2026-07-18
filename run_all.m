% run_all.m
% Main entry point. Run this to execute the full pipeline.
% Set the path below to wherever you cloned this repo.
projectRoot = fileparts(mfilename('fullpath'));
srcPath = fullfile(projectRoot, 'src');
helpersPath = fullfile(projectRoot, 'src', 'helpers');
addpath(srcPath); addpath(helpersPath);
cd(srcPath);
run('s1_prepare_data.m')
run('s2_build_representations.m')
run('s3_train_cnn.m')
run('s4_train_graphsage.m')
run('s5_evaluate_baseline.m')
run('s6_robustness_test.m')
run('s7_explainability.m')
fprintf("Done. Results saved to results/ folder.\n")