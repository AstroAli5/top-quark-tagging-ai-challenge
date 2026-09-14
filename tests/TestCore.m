function tests = TestCore
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    testCase.applyFixture(matlab.unittest.fixtures.PathFixture(root));
end

function testROCMatchesPairwiseDefinition(testCase)
    scores = [0.1 0.3 0.3 0.8 0.8 0.8 0.9];
    labels = [0 1 0 1 0 1 1];
    positives = scores(labels == 1).';
    negatives = scores(labels == 0);
    pairs = double(positives > negatives) + 0.5*double(positives == negatives);
    [tpr,fpr,auc] = computeROC(scores,labels);
    verifyEqual(testCase,auc,mean(pairs(:)),'AbsTol',1e-12);
    verifyEqual(testCase,[tpr(1) fpr(1) tpr(end) fpr(end)],[0 0 1 1]);
    [~,~,auc] = computeROC(ones(4,1),[1 1 0 0]);
    verifyEqual(testCase,auc,0.5);
    [~,~,auc] = computeROC([0 1],[0 1]);
    verifyEqual(testCase,auc,1);
    [~,~,auc] = computeROC([1 0],[0 1]);
    verifyEqual(testCase,auc,0);
end

function testROCRejectsInvalidLabels(testCase)
    verifyError(testCase,@() computeROC([0.1 0.2],[1 1]),'topquark:SingleClass');
    verifyError(testCase,@() computeROC([0.1 0.2],[0 2]),'topquark:InvalidLabels');
    verifyError(testCase,@() computeROC([0.1 0.2],1),'topquark:InvalidLabels');
end

function testWrappedGeometryAndSingleParticle(testCase)
    phi = [pi-0.01; -pi+0.01; pi-0.02];
    pt = [20;30;40]; eta = [0.1;0.1;0.12];
    fv = [pt.*cosh(eta),pt.*cos(phi),pt.*sin(phi),pt.*sinh(eta)];
    [features,A] = buildJetGraph(fv,6);
    verifyTrue(testCase,all(isfinite(features(:))));
    verifyLessThan(testCase,max(abs(features(:,2))),0.04);
    verifyEqual(testCase,A,A.');
    verifyFalse(testCase,any(diag(A)));
    img = buildJetImage(fv,32);
    verifyEqual(testCase,sum(expm1(img(:))),sum(pt),'AbsTol',1e-9);
    [~,singleA] = buildJetGraph(fv(1,:),6);
    verifyEqual(testCase,size(singleA),[1 1]);
    verifyFalse(testCase,any(singleA(:)));
end

function testNoiseIdentityAndMass(testCase)
    momentum = [3 4 5; 6 -2 4];
    mass = [2;3];
    fv = [sqrt(sum(momentum.^2,2)+mass.^2),momentum];
    before = rng;
    verifyEqual(testCase,injectDetectorNoise(fv,0),fv);
    verifyEqual(testCase,rng,before);
    rng(7);
    noisy = injectDetectorNoise(fv,0.1);
    verifyEqual(testCase,noisy(:,1).^2-sum(noisy(:,2:4).^2,2),mass.^2,'AbsTol',1e-10);
    verifyError(testCase,@() injectDetectorNoise(fv,-1),'MATLAB:expectedNonnegative');
end

function testStratifiedSplitsAreDisjointAndRepeatable(testCase)
    labels = [zeros(13,1); ones(17,1)];
    [a,b,c] = stratifiedSplit(labels,42);
    [a2,b2,c2] = stratifiedSplit(labels,42);
    verifyEqual(testCase,{a,b,c},{a2,b2,c2});
    verifyEqual(testCase,sort([a;b;c]),(1:30).');
    for indices = {a,b,c}
        verifyEqual(testCase,unique(labels(indices{1})),[0;1]);
    end
end

function testSparseBatchesAndGraphGradients(testCase)
    rng(123);
    features = {randn(3,4);randn(2,4);randn(1,4)};
    adjacency = {logical([0 1 1;1 0 1;1 1 0]);logical([0 1;1 0]);false(1)};
    [X,A,counts,T] = preprocessGraphMiniBatch(features,adjacency,[0;1;0]);
    verifyTrue(testCase,issparse(A));
    verifyEqual(testCase,full(A(1:3,4:end)),zeros(3,3));
    parameters = initializeGraphSAGE(4,8);
    [loss,grads] = dlfeval(@modelLossGraphSAGE,parameters,dlarray(X),A,counts,T);
    verifyTrue(testCase,isfinite(double(extractdata(loss))));
    for name = {'sage1','sage2','sage3','classify'}
        grad = extractdata(grads.(name{1}).Weights);
        verifyTrue(testCase,all(isfinite(grad(:))));
    end
    allAtOnce = predictGraphSAGE(parameters,features,adjacency,3);
    oneAtATime = predictGraphSAGE(parameters,features,adjacency,1);
    verifyEqual(testCase,oneAtATime,allAtOnce,'AbsTol',1e-10);
end

function testPoolPreservesGradients(testCase)
    [loss,grad] = dlfeval(@poolLoss,dlarray(reshape(1:10,5,2)));
    verifyTrue(testCase,isfinite(extractdata(loss)));
    expected = repmat([0.5;0.5;1/3;1/3;1/3],1,2);
    verifyEqual(testCase,extractdata(grad),expected,'AbsTol',1e-12);
end

function [loss,grad] = poolLoss(X)
    Y = globalMeanPool(X,[2;3]);
    loss = sum(Y,'all');
    grad = dlgradient(loss,X);
end

function testDatasetMismatchIsRejected(testCase)
    a = struct('datasetId','a');
    b = struct('datasetId','b');
    verifyError(testCase,@() verifyDatasetIds(a,b),'topquark:DatasetMismatch');
    verifyError(testCase,@() verifyDatasetIds(a,struct()),'topquark:DatasetMismatch');
end
