function tests = TestResNet18
    tests = functiontests(localfunctions);
end

function testNamedResidualNetworkOnGrayscaleJets(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    testCase.applyFixture(matlab.unittest.fixtures.PathFixture(root));
    testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,'src','bigdata')));
    cfg = project238Config; cfg.architecture = 'resnet18';
    savedRNG = rng; cleanup = onCleanup(@() rng(savedRNG)); rng(101);
    net = project238Network(cfg);
    verifyEqual(testCase,net.Layers(1).InputSize,[32 32 1]);
    residualBlocks = arrayfun(@(layer) isa(layer,'nnet.cnn.layer.AdditionLayer'),net.Layers);
    verifyEqual(testCase,sum(residualBlocks),8);
    images = rand(32,32,1,3,'single');
    scores = minibatchpredict(net,images,ExecutionEnvironment='cpu', ...
        MiniBatchSize=2,InputDataFormats='SSCB',OutputDataFormats='BC');
    if isa(scores,'dlarray'), scores = extractdata(scores); end
    verifySize(testCase,scores,[3 2]);
    verifyTrue(testCase,all(isfinite(scores),'all'));
    verifyEqual(testCase,sum(scores,2),ones(3,1,'like',scores),'AbsTol',single(1e-5));
end
