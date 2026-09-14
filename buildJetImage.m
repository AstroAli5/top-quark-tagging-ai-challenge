function image = buildJetImage(fourVectors,imageSize,etaRange,phiRange)
%BUILDJETIMAGE Convert one jet's particle four-vectors into a 2D
%pT-weighted "jet image" for a CNN — the standard jet-image convention
%from the jet-tagging literature (treating a jet as an image so ordinary
%2D convolutions can be applied to it).
%
%   IMAGE = BUILDJETIMAGE(FOURVECTORS,IMAGESIZE,ETARANGE,PHIRANGE)
%
%   FOURVECTORS - [numParticles x 4] matrix of (E, px, py, pz).
%   IMAGESIZE   - output image is IMAGESIZE x IMAGESIZE (default 32).
%   ETARANGE    - +/- window around the jet axis in eta (default 1.2).
%   PHIRANGE    - +/- window around the jet axis in phi (default 1.2).
%
%   IMAGE - [imageSize x imageSize] matrix. Each pixel holds the
%           log-compressed summed pT of every particle landing in that
%           (eta,phi) cell relative to the jet axis.

    if nargin < 2, imageSize = 32; end
    if nargin < 3, etaRange = 1.2; end
    if nargin < 4, phiRange = 1.2; end

    validateattributes(fourVectors,{'numeric'},{'2d','real','finite','nonempty','ncols',4});
    fourVectors = double(fourVectors);
    if any(fourVectors(:,1) <= 0 | hypot(fourVectors(:,2),fourVectors(:,3)) <= 0)
        error('topquark:InvalidParticles','Remove padding and require positive energy and pT.');
    end

    px = fourVectors(:,2);
    py = fourVectors(:,3);
    pz = fourVectors(:,4);

    pT  = sqrt(px.^2 + py.^2);
    phi = atan2(py,px);
    eta = asinh(pz./pT);

    jetEta = sum(pT.*eta)/sum(pT);
    jetPhi = atan2(sum(pT.*sin(phi)),sum(pT.*cos(phi)));

    dEta = eta - jetEta;
    dPhi = mod(phi - jetPhi + pi, 2*pi) - pi;

    etaEdges = linspace(-etaRange,etaRange,imageSize+1);
    phiEdges = linspace(-phiRange,phiRange,imageSize+1);

    image = zeros(imageSize,imageSize);
    etaBin = discretize(dEta,etaEdges);
    phiBin = discretize(dPhi,phiEdges);

    valid = find(~isnan(etaBin) & ~isnan(phiBin))';
    for i = valid
        image(phiBin(i),etaBin(i)) = image(phiBin(i),etaBin(i)) + pT(i);
    end

    % log-compress: pT spans orders of magnitude, and without this a
    % handful of very hot pixels would swamp everything else.
    image = log1p(image);
end
