% run_all.m
% -------------------------------------------------------
% MAIN ENTRY POINT - runs the full pipeline end to end.
% Just press Run. Takes about 30-40 minutes total.
% -------------------------------------------------------
cd('/MATLAB Drive/top-quark-tagging-ai-challenge/src')
addpath('/MATLAB Drive/top-quark-tagging-ai-challenge/src/helpers')
fprintf("=== Top Quark Tagging Pipeline ===\n")
run('s1_prepare_data.m')
run('s2_build_representations.m')
run('s3_train_cnn.m')
run('s4_train_graphsage.m')
run('s5_evaluate_baseline.m')
run('s6_robustness_test.m')
run('s7_explainability.m')
fprintf("=== All done! Results saved to /MATLAB Drive/results/ ===\n")