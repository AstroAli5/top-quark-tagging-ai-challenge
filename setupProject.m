function setupProject
%SETUPPROJECT Add the implementation folders; public commands remain at root.
    root = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(root,'src')));
end
