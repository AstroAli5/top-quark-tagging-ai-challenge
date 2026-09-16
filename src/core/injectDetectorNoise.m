function noisyFourVectors = injectDetectorNoise(fourVectors,noiseLevel,stream)
%INJECTDETECTORNOISE Simulate imperfect detector resolution by adding
%proportional Gaussian smearing to each particle's momentum.
%
%   NOISYFOURVECTORS = INJECTDETECTORNOISE(FOURVECTORS,NOISELEVEL)
%
%   FOURVECTORS - [numParticles x 4] matrix of (E, px, py, pz).
%   NOISELEVEL  - fractional resolution, e.g. 0.05 = 5% smearing per
%                 particle per component. This is a deliberately simple
%                 stand-in for real detector effects (actual
%                 calorimeter/tracker resolution depends on energy,
%                 particle type, and detector region) — good enough to
%                 probe robustness in a controlled way, and it's called
%                 out as a simplification in the README rather than
%                 presented as a full detector simulation.
%
%   Each momentum component is scaled by (1 + noiseLevel*randn),
%   independently per particle and per component. Energy is then
%   recomputed from the smeared momentum so the particle stays physical
%   (keeps the original mass fixed: E^2 - |p|^2 = m^2). NOISELEVEL = 0
%   returns the input completely unchanged.

    validateattributes(noiseLevel,{'numeric'},{'scalar','real','finite','nonnegative'});
    if noiseLevel == 0
        noisyFourVectors = fourVectors;
        return;
    end

    E  = fourVectors(:,1);
    px = fourVectors(:,2);
    py = fourVectors(:,3);
    pz = fourVectors(:,4);

    numParticles = size(fourVectors,1);
    if nargin < 3
        draws = randn(numParticles,3);
    else
        draws = randn(stream,numParticles,3);
    end
    smear = 1 + noiseLevel*draws;

    pxNoisy = px .* smear(:,1);
    pyNoisy = py .* smear(:,2);
    pzNoisy = pz .* smear(:,3);

    massSq = max(E.^2 - (px.^2 + py.^2 + pz.^2), 0);
    ENoisy = sqrt(pxNoisy.^2 + pyNoisy.^2 + pzNoisy.^2 + massSq);

    noisyFourVectors = [ENoisy, pxNoisy, pyNoisy, pzNoisy];
end
