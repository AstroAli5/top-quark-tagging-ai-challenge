function tests = TestWinnerFeatures
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    testCase.applyFixture(matlab.unittest.fixtures.PathFixture(root));
end

function testOccupiedMomentsAndEmptyJet(testCase)
    cfg = projectConfig;
    [image,features] = buildWinnerJetFeatures([10 10 0 0;20 20 0 0],cfg);
    pixel = reshape(image(19,19,:),1,[]);
    verifyEqual(testCase,pixel(1:6),single([2 30 0 30 25 25]));
    verifyEqual(testCase,pixel(7:11),single([0 0 1 1 15]));
    verifyTrue(testCase,all(isfinite(features)));
    [emptyImage,emptyFeatures] = buildWinnerJetFeatures(zeros(0,4),cfg);
    verifyEqual(testCase,sum(abs(emptyImage(:))),single(0));
    verifyEqual(testCase,emptyFeatures,zeros(1,4,'single'));
    [singleImage,singleFeatures] = buildWinnerJetFeatures([10 10 0 0],cfg);
    verifyTrue(testCase,all(isfinite(singleImage(:))) && all(isfinite(singleFeatures)));
end

function testAnglesParticleOrderAndEnergy(testCase)
    cfg = projectConfig;
    pt = [50;30;20]; eta = [0.1;0.3;0.15]; phi = [pi-0.02;-pi+0.02;pi-0.05];
    fv = [pt.*cosh(eta),pt.*cos(phi),pt.*sin(phi),pt.*sinh(eta)];
    [image,features] = buildWinnerJetFeatures(fv,cfg);
    [permuted,radial] = buildWinnerJetFeatures(fv([3 1 2],:),cfg);
    verifyEqual(testCase,permuted,image);
    verifyEqual(testCase,radial,features);
    verifyEqual(testCase,double(sum(image(:,:,4),'all')),sum(fv(:,1)),'AbsTol',1e-4);
    verifyEqual(testCase,sum(image(:,:,1),'all'),single(3));
    % A common azimuthal rotation, including crossing +/-pi, is immaterial.
    phi = phi+0.4;
    fv(:,2:3) = [pt.*cos(phi),pt.*sin(phi)];
    [rotated,rotatedFeatures] = buildWinnerJetFeatures(fv,cfg);
    verifyEqual(testCase,rotated,image,'AbsTol',single(1e-4));
    verifyEqual(testCase,rotatedFeatures,features,'AbsTol',single(1e-5));
end

function testTrainingStatisticsAndFrozenInference(testCase)
    cfg = projectConfig;
    jets = {[10 10 0 0];[20 20 0 0]};
    normalization = fitWinnerNormalization(jets,cfg);
    expectedMean = (log1p(10)+log1p(20))/(2*37*37);
    verifyEqual(testCase,double(normalization.imageMean(1,1,4)),expectedMean,'AbsTol',1e-9);
    before = normalization;
    extreme = winnerObservation([1e6 1e6 0 0],normalization,cfg);
    verifyTrue(testCase,all(isfinite(extreme{1}(:))));
    verifyEqual(testCase,normalization,before);
    verifyEqual(testCase,normalization.trainingJets,2);
end
