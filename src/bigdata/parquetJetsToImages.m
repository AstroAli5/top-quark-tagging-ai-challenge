function manifest = parquetJetsToImages(cfg)
%PARQUETJETSTOIMAGES Evaluate a tall transform directly into labelled TIFFs.
    parquetRoot = fullfile(cfg.dataDir,'parquet');
    manifest = jsondecode(fileread(fullfile(parquetRoot,'manifest.json')));
    imageRoot = fullfile(cfg.dataDir,'images');
    if ~isfolder(imageRoot), mkdir(imageRoot); end
    for name = ["train","val","test"]
        expected = manifest.partitions.(name);
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
            write(folder,images,WriteFcn=@writeJetImageBlock);
            writeProjectJSON(done,identity);
        end
        jetImageDatastore(folder,expected,cfg.batchSize);
        fprintf('%s: verified %d labelled image files.\n',name,expected.selected_rows);
    end
end
