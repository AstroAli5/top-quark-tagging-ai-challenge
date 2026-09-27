function writeJetImageBlock(info,block)
%WRITEJETIMAGEBLOCK Tall writer. Source IDs make filenames unique and stable.
% Only write under RequiredLocation, so tall/write can commit its temporary data.
    for label = ["background","signal"]
        folder = fullfile(info.RequiredLocation,label);
        if ~isfolder(folder), mkdir(folder); end
    end
    for j = 1:height(block)
        if ~ismember(block.Label(j),[0 1])
            error('topquark:InvalidLabels','Only binary labels are supported.');
        end
        names = ["background","signal"];
        filename = fullfile(info.RequiredLocation,names(double(block.Label(j))+1), ...
            sprintf('jet_%09d.tif',block.SourceRow(j)));
        pixels = block.Image{j};
        t = Tiff(filename,'w');
        cleanup = onCleanup(@() close(t));
        tags = struct('ImageLength',size(pixels,1),'ImageWidth',size(pixels,2), ...
            'Photometric',Tiff.Photometric.MinIsBlack,'BitsPerSample',32, ...
            'SamplesPerPixel',1,'RowsPerStrip',size(pixels,1), ...
            'PlanarConfiguration',Tiff.PlanarConfiguration.Chunky, ...
            'SampleFormat',Tiff.SampleFormat.IEEEFP,'Compression',Tiff.Compression.LZW);
        setTag(t,tags); write(t,pixels); clear cleanup;
    end
end
