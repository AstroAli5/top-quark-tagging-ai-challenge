function s4_train_graphsage(cfg)
%S4_TRAIN_GRAPHSAGE Sparse CPU GraphSAGE training with batched validation.
    if nargin < 1, cfg = projectConfig; end
    fprintf('Step 4/7: Training GraphSAGE...\n');
    data = load(fullfile(cfg.dataDir,'jet_graphs.mat'));
    rng(cfg.graphSeed,'twister');
    parameters = initializeGraphSAGE(size(data.jetNodeFeatures{1},2),cfg.hiddenSize);
    trailingAvg = []; trailingAvgSq = [];
    iteration = 0;
    bestLoss = Inf;
    bestEpoch = 0;
    history = zeros(cfg.graphEpochs,2);
    for epoch = 1:cfg.graphEpochs
        order = data.idxTrain(randperm(numel(data.idxTrain)));
        lossSum = 0;
        count = 0;
        for first = 1:cfg.graphBatchSize:numel(order)
            idx = order(first:min(first+cfg.graphBatchSize-1,numel(order)));
            [X,A,numNodes,T] = preprocessGraphMiniBatch( ...
                data.jetNodeFeatures(idx),data.jetAdjacency(idx),data.labels(idx));
            [loss,gradients] = dlfeval(@modelLossGraphSAGE,parameters,dlarray(X),A,numNodes,T);
            scalarLoss = double(extractdata(loss));
            if ~isfinite(scalarLoss)
                error('topquark:NonfiniteLoss','GraphSAGE loss is not finite.');
            end
            iteration = iteration + 1;
            [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
                trailingAvg,trailingAvgSq,iteration,cfg.graphLearnRate);
            lossSum = lossSum + scalarLoss*numel(idx);
            count = count + numel(idx);
        end
        p = predictGraphSAGE(parameters,data.jetNodeFeatures(data.idxVal), ...
            data.jetAdjacency(data.idxVal),cfg.graphBatchSize);
        targets = data.labels(data.idxVal);
        p = min(max(p,1e-12),1-1e-12);
        valLoss = -mean(targets.*log(p)+(1-targets).*log1p(-p));
        history(epoch,:) = [lossSum/count valLoss];
        if valLoss < bestLoss
            bestLoss = valLoss;
            bestParameters = parameters;
            bestEpoch = epoch;
        end
        fprintf('  Epoch %d/%d: loss %.4f, validation loss %.4f\n', ...
            epoch,cfg.graphEpochs,history(epoch,1),valLoss);
    end
    parameters = bestParameters;
    datasetId = data.datasetId;
    trainingHistory = array2table(history,VariableNames={'TrainingLoss','ValidationLoss'});
    if ~isfolder(cfg.modelsDir), mkdir(cfg.modelsDir); end
    save(fullfile(cfg.modelsDir,'graphsage_model.mat'), ...
        'parameters','datasetId','cfg','trainingHistory','bestEpoch','bestLoss');
    fprintf('Saved GraphSAGE model from epoch %d.\n',bestEpoch);
end
