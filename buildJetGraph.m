function [nodeFeatures,adjacency] = buildJetGraph(fourVectors,k)
%BUILDJETGRAPH Convert one jet's particle four-vectors into a graph.
%
%   [NODEFEATURES,ADJACENCY] = BUILDJETGRAPH(FOURVECTORS,K)
%
%   FOURVECTORS - [numParticles x 4] matrix of (E, px, py, pz) for every
%                 non-zero-padded particle in one jet.
%   K           - number of nearest neighbors to connect each particle to
%                 (default 6), measured by angular distance deltaR in
%                 (eta,phi) space. This is the standard "particle cloud"
%                 graph-construction convention used in jet-tagging
%                 literature (e.g. ParticleNet, Qu & Gouskos 2020).
%
%   NODEFEATURES - [numParticles x 4] matrix of (deltaEta, deltaPhi,
%                  log(pT), log(E)), relative to the jet axis.
%   ADJACENCY    - [numParticles x numParticles] symmetric 0/1 adjacency
%                  matrix, NO self-loops.

    if nargin < 2
        k = 6;
    end

    E  = fourVectors(:,1);
    px = fourVectors(:,2);
    py = fourVectors(:,3);
    pz = fourVectors(:,4);

    pT  = sqrt(px.^2 + py.^2);
    p   = sqrt(px.^2 + py.^2 + pz.^2);
    phi = atan2(py,px);
    eta = 0.5*log((p + pz + eps)./(p - pz + eps));

    % Jet axis: the pT-weighted average direction of all particles in it.
    jetEta = sum(pT.*eta)/sum(pT);
    jetPhi = atan2(sum(pT.*sin(phi)),sum(pT.*cos(phi)));

    dEta = eta - jetEta;
    dPhi = wrapPhi(phi - jetPhi);

    nodeFeatures = [dEta, dPhi, log(pT + 1e-6), log(E + 1e-6)];

    % k-nearest-neighbor graph by angular distance deltaR = sqrt(dEta^2+dPhi^2)
    numParticles = size(fourVectors,1);
    kUse = min(k,numParticles-1);

    deltaEtaMat = eta - eta.';
    deltaPhiMat = wrapPhi(phi - phi.');
    deltaR = sqrt(deltaEtaMat.^2 + deltaPhiMat.^2);
    deltaR(1:numParticles+1:end) = Inf; % exclude self from nearest-neighbor search

    adjacency = false(numParticles,numParticles);
    for i = 1:numParticles
        [~,idx] = mink(deltaR(i,:),kUse);
        adjacency(i,idx) = true;
    end
    adjacency = adjacency | adjacency.'; % symmetrize
end

function wrapped = wrapPhi(angle)
%WRAPPHI Wrap an angle (or matrix of angles) to the range [-pi, pi].
%   Implemented directly with mod() rather than relying on Mapping
%   Toolbox's wrapToPi, so this only needs base MATLAB.
    wrapped = mod(angle + pi, 2*pi) - pi;
end
