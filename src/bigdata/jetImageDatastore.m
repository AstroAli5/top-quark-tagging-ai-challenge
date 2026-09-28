function [ds,rows] = jetImageDatastore(folder,expected,batchSize)
%JETIMAGEDATASTORE Validate coverage, then order files by source ordinal.
    timer = tic;
    fprintf('Opening image datastore for %d source rows.\n',expected.selected_rows);
    ds = imageDatastore(folder,IncludeSubfolders=true, ...
        LabelSource='foldernames',FileExtensions={'.tif'});
    % Files is a datastore property; retrieve the complete list only once.
    files = ds.Files;
    fprintf('Listed %d image files in %.1f seconds.\n',numel(files),toc(timer));
    rows = zeros(numel(files),1);
    for j = 1:numel(files)
        [~,name] = fileparts(files{j});
        token = regexp(name,'^jet_(\d{9})$','tokens','once');
        if isempty(token), error('topquark:ImageCoverage','Unexpected image filename.'); end
        rows(j) = str2double(token{1});
        if mod(j,100000)==0
            fprintf('Parsed %d / %d source IDs in %.1f seconds.\n',j,numel(files),toc(timer));
        end
    end
    [rows,order] = sort(rows);
    if ~isequal(rows,(0:expected.selected_rows-1).')
        error('topquark:ImageCoverage','Missing, duplicate or out-of-range source row.');
    end
    ds = subset(ds,order);
    names = ["background","signal"];
    counts = [sum(ds.Labels==names(1)),sum(ds.Labels==names(2))];
    if ~isequal(counts,double(expected.class_counts(:).')) || any(isundefined(ds.Labels))
        error('topquark:ImageLabels','Image folder labels disagree with source counts.');
    end
    ds.ReadSize = batchSize;
    fprintf('Datastore coverage and labels verified in %.1f seconds.\n',toc(timer));
end
