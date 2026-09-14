function verifyDatasetIds(varargin)
%VERIFYDATASETIDS Reject stale models or representations from another import.
    for i = 1:nargin
        if ~isfield(varargin{i},'datasetId') || ...
                ~strcmp(varargin{1}.datasetId,varargin{i}.datasetId)
            error('topquark:DatasetMismatch', ...
                'Models and data do not share a dataset ID. Rerun steps 1-4 together.');
        end
    end
end
