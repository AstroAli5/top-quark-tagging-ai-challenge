function [summary,paired] = summarize_matlab(inputDir,outputDir)
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
    fig = figure('Visible','off','Position',[100 100 1500 500]);
    cleanup = onCleanup(@() close(fig));
    tiledlayout(1,3,TileSpacing='compact',Padding='compact');
    first = nexttile; hold(first,'on');
    second = nexttile; hold(second,'on');
    third = nexttile; hold(third,'on');
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
        errorbar(first,j,100*meanAccuracy,100*sdAccuracy,'o',LineWidth=1.5,DisplayName=names(j));
        errorbar(second,j,meanAUC,sdAUC,'o',LineWidth=1.5,DisplayName=names(j));
        current = noisePerSeed(noisePerSeed.Model==names(j),:);
        stats = groupsummary(current,'Sigma',{'mean','std'},'mean_AUC');
        errorbar(third,stats.Sigma,stats.mean_mean_AUC,stats.std_mean_AUC, ...
            '-o',LineWidth=1.5,DisplayName=names(j));
    end
    xticks(first,1:numel(names)); xticklabels(first,names); xtickangle(first,20);
    xlim(first,[.5 numel(names)+.5]);
    ylabel(first,'Accuracy (%)'); title(first,'Clean accuracy: mean ± seed SD'); grid(first,'on');
    xticks(second,1:numel(names)); xticklabels(second,names); xtickangle(second,20);
    xlim(second,[.5 numel(names)+.5]);
    ylabel(second,'Clean AUC'); title(second,'Clean AUC: mean ± seed SD'); grid(second,'on');
    xlabel(third,'Synthetic component smearing'); ylabel(third,'Noise-sample AUC');
    title(third,'Noise: mean ± seed SD'); grid(third,'on');
    key = legend(third,Orientation='horizontal'); key.Layout.Tile = 'south';
    paired = table;
    for a = 1:numel(names)-1
        left = sortrows(clean(clean.Model==names(a),:),'Seed');
        for b = a+1:numel(names)
            right = sortrows(clean(clean.Model==names(b),:),'Seed');
            if ~isequal(left.Seed,right.Seed)
                error('topquark:UnpairedSeeds','Paired comparisons require identical seed labels.');
            end
            differences = [right.Accuracy-left.Accuracy,right.AUC-left.AUC];
            n = height(left); df = n-1;
            critical = sqrt(df*(1/betaincinv(.05,df/2,.5)-1));
            means = mean(differences,1); deviations = std(differences,0,1);
            half = critical*deviations/sqrt(n);
            row = table(names(b)+" minus "+names(a),n,means(1),deviations(1), ...
                means(1)-half(1),means(1)+half(1),means(2),deviations(2), ...
                means(2)-half(2),means(2)+half(2),VariableNames={ ...
                'Difference','Seeds','MeanAccuracyDifference','AccuracySeedSD', ...
                'AccuracySeedCI_Low','AccuracySeedCI_High','MeanAUCDifference', ...
                'AUCSeedSD','AUCSeedCI_Low','AUCSeedCI_High'});
            paired = [paired;row]; %#ok<AGROW>
        end
    end
    sgtitle('MATLAB analysis of the recorded study (training subsets)',FontSize=14);
    exportgraphics(fig,fullfile(outputDir,'matlab_study_summary.png'),Resolution=180);
    writetable(summary,fullfile(outputDir,'clean_summary_matlab.csv'));
    writetable(noisePerSeed,fullfile(outputDir,'noise_per_seed_matlab.csv'));
    writetable(paired,fullfile(outputDir,'paired_summary_matlab.csv'));
    disp(summary);
    disp(paired);
end
