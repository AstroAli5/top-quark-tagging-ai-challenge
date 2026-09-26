function [ds,rows] = jetImageDatastore(folder,expected,batchSize)
%JETIMAGEDATASTORE Validate coverage, then order files by source ordinal.
    ds = imageDatastore(folder,IncludeSubfolders=true, ...
        LabelSource='foldernames',FileExtensions={'.tif'});
    rows = zeros(numel(ds.Files),1);
    for j = 1:numel(ds.Files)
        [~,name] = fileparts(ds.Files{j});
        token = regexp(name,'^jet_(\d{9})$','tokens','once');
        if isempty(token), error('topquark:ImageCoverage','Unexpected image filename.'); end
        rows(j) = str2double(token{1});
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
end
