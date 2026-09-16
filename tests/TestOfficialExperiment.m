function tests = TestOfficialExperiment
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    testCase.applyFixture(matlab.unittest.fixtures.PathFixture(root));
    for folder = {'core','pipeline','reference','experiment'}
        testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,'src',folder{1})));
    end
end

function testOfficialPartitionsAndSavedPredictions(testCase)
    folder = tempname; mkdir(folder);
    cleanup = onCleanup(@() rmdir(folder,'s'));
    official = fullfile(folder,'official'); mkdir(official);
    rng(99);
    labels = repmat([0;1],30,1);
    particleData = zeros(60,800,'single');
    for j = 1:60
        pt = 4+20*rand(6,1); eta = 0.2*randn(6,1); phi = 0.3*randn(6,1);
        fv = [pt.*cosh(eta),pt.*cos(phi),pt.*sin(phi),pt.*sinh(eta)];
        particleData(j,1:24) = reshape(fv.',1,[]);
    end
    partition = [ones(40,1);2*ones(20,1)];
    provenance_json = jsonencode(struct('source','Synthetic official-partition fixture'));
    save(fullfile(official,'fitting.mat'),'particleData','labels','partition','provenance_json');
    manifest = struct('dataset','10.5281/zenodo.2603256','train_count',40,'val_count',20, ...
        'test_count',20,'full_official_test',false,'test_chunks',[]);
    for c = 1:2
        sourceRows = (10*(c-1):10*c-1).';
        labels = repmat([0;1],5,1);
        particleData = circshift(particleData(1:10,:),c,1);
        name = sprintf('test_%d.mat',c);
        save(fullfile(official,name),'particleData','labels','sourceRows');
        manifest.test_chunks = [manifest.test_chunks;struct('file',name,'start',10*(c-1),'stop',10*c)];
    end
    fid = fopen(fullfile(official,'manifest.json'),'w'); fprintf(fid,'%s',jsonencode(manifest)); fclose(fid);
    output = fullfile(folder,'run');
    overrides = struct('cnnEpochs',1,'graphEpochs',1,'winnerEpochs',1,'hiddenSize',8, ...
        'winnerWidths',[8 16],'winnerGroups',2,'cnnBatchSize',8,'graphBatchSize',7, ...
        'winnerBatchSize',7,'noiseTestJets',14,'noiseLevels',[0 0.1],'noiseSeeds',[7 17]);
    run_experiment(101,official,output,overrides);
    split = load(fullfile(output,'data','jet_split.mat'));
    verifyEqual(testCase,split.idxTrain,(1:40).');
    verifyEqual(testCase,split.idxVal,(41:60).');
    verifyEmpty(testCase,split.idxTest);
    clean = load(fullfile(output,'results','clean_predictions.mat'));
    noisy = load(fullfile(output,'results','noise_predictions.mat'));
    verifyEqual(testCase,clean.rows,(0:19).');
    for r = 1:2
        verifyEqual(testCase,noisy.noisePredictions(:,:,1,r),single(clean.probabilities(1:14,:)));
    end
    verifyFalse(testCase,isequal(noisy.noisePredictions(:,:,2,1),noisy.noisePredictions(:,:,2,2)));
    trained = load(fullfile(output,'models','cnn_model.mat'),'cfg');
    cfg = trained.cfg;
    for selected = {{'CNN','GraphSAGE'},{'ResNeXt-SE reference'}}
        cfg.experimentModels = selected{1};
        cfg.resultsDir = tempname(folder);
        evaluateOfficialTest(cfg,official,manifest,101,zeros(1,numel(selected{1})));
        separate = load(fullfile(cfg.resultsDir,'clean_predictions.mat'));
        columns = ismember(["CNN","GraphSAGE","ResNeXt-SE reference"],string(selected{1}));
        verifyEqual(testCase,separate.probabilities,clean.probabilities(:,columns));
    end
    verifyError(testCase,@() run_experiment(101,official,output,overrides),'topquark:ExistingRun');
end

function testCachedFeaturesMatchStreamed(testCase)
    cfg = projectConfig; jets = {[10 10 0 0;5 4 3 0;3 2 2 1]};
    normalization = fitWinnerNormalization(jets,cfg);
    streamed = winnerDatastore(jets,1,normalization,cfg);
    cached = cachedWinnerDatastore(jets,1,normalization,cfg);
    verifyEqual(testCase,read(cached),read(streamed));
    before = rng;
    a = RandStream('mt19937ar','Seed',7); b = RandStream('mt19937ar','Seed',7);
    verifyEqual(testCase,injectDetectorNoise(jets{1},0.1,a),injectDetectorNoise(jets{1},0.1,b));
    verifyEqual(testCase,rng,before);
end

function testSparseConstituentTestJetsAreRetained(testCase)
    path = [tempname '.mat']; cleanup = onCleanup(@() delete(path));
    particleData = zeros(2,800,'single');
    particleData(1,1:4) = [10 10 0 0];
    particleData(2,1:8) = [10 10 0 0 5 4 3 0];
    labels = [0;1]; sourceRows = [49496;49497];
    save(path,'particleData','labels','sourceRows');
    [jets,y,rows] = readJetChunk(path);
    verifyEqual(testCase,cellfun(@(x) size(x,1),jets),[1;2]);
    verifyEqual(testCase,y,labels); verifyEqual(testCase,rows,sourceRows);
    for j = 1:2
        [features,adjacency] = buildJetGraph(jets{j},6);
        verifyEqual(testCase,size(features),[j 4]);
        verifySize(testCase,adjacency,[j j]);
        verifyTrue(testCase,all(isfinite(buildJetImage(jets{j})),'all'));
    end
    particleData(1,:) = 0; save(path,'particleData','labels','sourceRows');
    verifyError(testCase,@() readJetChunk(path),'topquark:InvalidTestJet');
end
