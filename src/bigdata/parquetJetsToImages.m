function manifest = parquetJetsToImages(cfg)
%PARQUETJETSTOIMAGES Evaluate a tall transform directly into labelled TIFFs.
    parquetRoot = fullfile(cfg.dataDir,'parquet');
    manifest = jsondecode(fileread(fullfile(parquetRoot,'manifest.json')));
    imageRoot = fullfile(cfg.dataDir,'images');
    if ~isfolder(imageRoot), mkdir(imageRoot); end
    for name = ["train","val","test"]
        timer = tic;
        expected = manifest.partitions.(name);
        progressPath = fullfile(cfg.dataDir,'image_progress.json');
        progress = struct('partition',char(name),'expectedRows',expected.selected_rows, ...
            'stage','checking_cache','elapsedSeconds',0);
        writeProjectJSON(progressPath,progress);
        folder = fullfile(imageRoot,name);
        done = fullfile(imageRoot,name+".json");
        identity = struct('parquetManifestSHA256',projectFileSHA256(fullfile(parquetRoot,name,'manifest.json')), ...
            'imageSize',cfg.imageSize,'imageBuilderSHA256',projectFileSHA256(which('buildJetImage')), ...
            'transformSHA256',projectFileSHA256(which('jetTableToImages')));
        if isfile(done)
            previous = jsondecode(fileread(done));
            if ~isequal(previous,identity)
                error('topquark:ImageCacheMismatch','Choose a new dataDir for changed preprocessing.');
            end
        else
            if isfolder(folder)
                error('topquark:IncompleteImages','Incomplete image folder; use a new dataDir.');
            end
            pds = parquetDatastore(fullfile(parquetRoot,name,'*.parquet'));
            pds.ReadSize = cfg.chunkRows;
            tt = tall(pds);
            prototype = table(int64(0),int8(0),{zeros(cfg.imageSize,'single')}, ...
                VariableNames={'SourceRow','Label','Image'});
            images = matlab.tall.transform(@(block) jetTableToImages(block,cfg.imageSize), ...
                tt,OutputsLike={prototype});
            progress.stage = 'writing_images'; writeProjectJSON(progressPath,progress);
            write(folder,images,WriteFcn=@writeJetImageBlock);
            fprintf('%s: image write returned after %.1f seconds.\n',name,toc(timer));
            writeProjectJSON(done,identity);
        end
        progress.stage = 'validating_datastore'; progress.elapsedSeconds = toc(timer);
        writeProjectJSON(progressPath,progress);
        jetImageDatastore(folder,expected,cfg.batchSize);
        progress.stage = 'complete'; progress.elapsedSeconds = toc(timer);
        writeProjectJSON(progressPath,progress);
        fprintf('%s: verified %d labelled image files.\n',name,expected.selected_rows);
    end
end
