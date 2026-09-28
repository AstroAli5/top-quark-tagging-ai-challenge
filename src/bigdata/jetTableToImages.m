function output = jetTableToImages(input,imageSize)
%JETTABLETOIMAGES Pure block transform; never loads the complete tall table.
    columns = cell(1,800);
    components = {'E','PX','PY','PZ'};
    for p = 0:199
        for c = 1:4, columns{4*p+c} = sprintf('%s_%d',components{c},p); end
    end
    values = input{:,columns}; % one bounded block; avoid 800 table lookups per jet
    pixels = cell(height(input),1);
    for i = 1:height(input)
        particles = reshape(double(values(i,:)),4,[]).';
        if any(~isfinite(particles),'all') || any(particles(:,1) < 0)
            error('topquark:InvalidParticles','Invalid values at source row %d.',input.SourceRow(i));
        end
        particles = particles(particles(:,1)>0 & hypot(particles(:,2),particles(:,3))>0,:);
        if isempty(particles)
            error('topquark:InvalidTestJet','Empty jet at source row %d.',input.SourceRow(i));
        end
        pixels{i} = single(buildJetImage(particles,imageSize));
    end
    output = table(int64(input.SourceRow),int8(input.Label),pixels, ...
        VariableNames={'SourceRow','Label','Image'});
end
