function summary = summarize_matlab(inputDir,outputDir)
%SUMMARIZE_MATLAB Recreate study tables and error-bar figures in MATLAB.
% Noise repeats are averaged within training seed before seed statistics.
    setupProject;
    root = fileparts(mfilename('fullpath'));
    if nargin < 1, inputDir = fullfile(root,'experiments','official-study'); end
    if nargin < 2, outputDir = fullfile(root,'results','matlab-summary'); end
    if ~isfolder(outputDir), mkdir(outputDir); end
    clean = readtable(fullfile(inputDir,'per_seed_clean.csv'),TextType='string');
    noise = readtable(fullfile(inputDir,'per_seed_noise.csv'),TextType='string');
    names = unique(clean.Model,'stable'); summary = table;
    fig = figure('Visible','off','Position',[100 100 1100 500]);
    cleanup = onCleanup(@() close(fig));
    tiledlayout(1,2,TileSpacing='compact',Padding='compact');
    first = nexttile; hold(first,'on');
    second = nexttile; hold(second,'on');
    noisePerSeed = groupsummary(noise,{'Model','Seed','Sigma'},'mean',{'Accuracy','AUC'});
    for j = 1:numel(names)
        c = clean(clean.Model==names(j),:);
        assert(numel(unique(c.Seed))==height(c) && height(c)>=2,'Expected distinct seeds.');
        n = height(c); df = n-1;
        % Two-sided Student-t 95% quantile, without an extra toolbox.
        critical = sqrt(df*(1/betaincinv(.05,df/2,.5)-1));
        meanAccuracy = mean(c.Accuracy); meanAUC = mean(c.AUC);
        sdAccuracy = std(c.Accuracy); sdAUC = std(c.AUC);
        half = critical*[sdAccuracy,sdAUC]/sqrt(n);
        row = table(names(j),n,meanAccuracy,sdAccuracy,meanAUC,sdAUC, ...
            meanAccuracy-half(1),meanAccuracy+half(1),meanAUC-half(2),meanAUC+half(2), ...
            VariableNames={'Model','Seeds','AccuracyMean','AccuracySeedSD','AUCMean','AUCSeedSD', ...
            'AccuracySeedCI_Low','AccuracySeedCI_High','AUCSeedCI_Low','AUCSeedCI_High'});
        summary = [summary;row]; %#ok<AGROW>
        errorbar(first,j,meanAUC,sdAUC,'o',LineWidth=1.5,DisplayName=names(j));
        current = noisePerSeed(noisePerSeed.Model==names(j),:);
        stats = groupsummary(current,'Sigma',{'mean','std'},'mean_AUC');
        errorbar(second,stats.Sigma,stats.mean_mean_AUC,stats.std_mean_AUC, ...
            '-o',LineWidth=1.5,DisplayName=names(j));
    end
    xticks(first,1:numel(names)); xticklabels(first,names); xtickangle(first,20);
    xlim(first,[.5 numel(names)+.5]);
    ylabel(first,'Clean AUC'); title(first,'Clean test: mean ± seed SD'); grid(first,'on');
    xlabel(second,'Synthetic component smearing'); ylabel(second,'Noise-sample AUC');
    title(second,'Noise: mean ± seed SD'); grid(second,'on');
    key = legend(second,Orientation='horizontal'); key.Layout.Tile = 'south';
    sgtitle('MATLAB analysis of the recorded study (training subsets)',FontSize=14);
    exportgraphics(fig,fullfile(outputDir,'matlab_study_summary.png'),Resolution=180);
    writetable(summary,fullfile(outputDir,'clean_summary_matlab.csv'));
    writetable(noisePerSeed,fullfile(outputDir,'noise_per_seed_matlab.csv'));
    disp(summary);
end
