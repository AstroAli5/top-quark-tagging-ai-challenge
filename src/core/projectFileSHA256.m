function value = projectFileSHA256(path)
%PROJECTFILESHA256 Bounded file hashing. MATLAB must run with its JVM enabled.
    if ~usejava('jvm')
        error('topquark:JVMRequired','SHA-256 provenance requires MATLAB with its JVM enabled.');
    end
    digest = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(path,'rb');
    if fid < 0, error('topquark:ReadFailed','Cannot read %s.',path); end
    cleanup = onCleanup(@() fclose(fid));
    while true
        bytes = fread(fid,1024*1024,'*int8');
        if isempty(bytes), break; end
        digest.update(bytes);
    end
    value = lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2).',1,[]));
end
