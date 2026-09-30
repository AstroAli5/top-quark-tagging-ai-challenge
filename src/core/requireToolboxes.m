function requireToolboxes(action,hasDeepLearning,oldRelease)
%REQUIRETOOLBOXES Fail early, once, with an actionable environment message.
%   REQUIRETOOLBOXES(ACTION) errors unless this session can run the saved
%   models. ACTION names the caller's work and appears in the message.
%
%   Call this before any load() of a checkpoint. MATLAB does not fail when
%   it reads a saved dlnetwork without Deep Learning Toolbox: it warns and
%   substitutes a uint32 placeholder, so the run continues past a verified
%   checksum and only breaks several frames later inside minibatchpredict.
%
%   The second and third arguments override the environment probes and
%   exist so both branches stay testable on any installation.
    if nargin < 1 || strlength(string(action)) == 0, action = "run this workflow"; end
    if nargin < 2, hasDeepLearning = ~isempty(ver('nnet')); end
    if nargin < 3, oldRelease = isMATLABReleaseOlderThan('R2024a'); end
    if oldRelease
        error('topquark:Requirements', ...
            ['MATLAB R2024a or later is required to %s; this session reports %s.\n' ...
             'The saved models use the trainnet/minibatchpredict interface introduced in R2024a.'], ...
            action,version('-release'));
    end
    if ~hasDeepLearning
        error('topquark:Requirements', ...
            ['Deep Learning Toolbox is required to %s.\n' ...
             'Without it MATLAB reads each saved dlnetwork as a uint32 placeholder ' ...
             'instead of a network, so checkpoint checksums still pass but no ' ...
             'prediction can be made.\n' ...
             'To review the recorded study without any toolbox, run: summarize_matlab'], ...
            action);
    end
end
