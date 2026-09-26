function writeProjectJSON(path,value)
    fid = fopen(path,'w');
    if fid < 0, error('topquark:WriteFailed','Cannot write %s.',path); end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
