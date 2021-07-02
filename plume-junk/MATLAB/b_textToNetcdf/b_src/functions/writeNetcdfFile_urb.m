function writeNetcdfFile_urb(outputDir,outputBaseName,  xCellGrid_urb,yCellGrid_urb,zCellGrid_urb, uMean,vMean,wMean)

    % verify the outputDir is a valid path
    if exist(outputDir, 'dir') ~= 7
        error('!!! writeNetcdfOutput_turb error !!! input outputDir \"%s\" does not exist or is not a valid directory!',outputDir);
    end
    
    % set the output filename
    outputFileName = sprintf('%s/%s_urb.nc',outputDir,outputBaseName);
    
    %%%% if the file already exists, delete it
    if exist(outputFileName, 'file') == 2
        delete(outputFileName);
    end
    
    % setup some other side variables
    nx = length(xCellGrid_urb);
    ny = length(yCellGrid_urb);
    nz = length(zCellGrid_urb);
    
    % don't worry if the output already exists, just overwrite it
    %%%save(outputFileName,  'xCellGrid_urb','yCellGrid_urb','zCellGrid_urb', 'uMean','vMean','wMean');
    times = 0;  % in case need more times later
    nccreate(outputFileName,'t','Dimensions',{'t',1,Inf},'Format','netcdf4');
    ncwrite(outputFileName,'t',times);
    ncwriteatt(outputFileName,'t','units','s');
    ncwriteatt(outputFileName,'t','long_name','time');
    nccreate(outputFileName,'x','Dimensions',{'x',nx});
    ncwrite(outputFileName,'x',xCellGrid_urb);
    ncwriteatt(outputFileName,'x','units','m');
    ncwriteatt(outputFileName,'x','long_name','x-distance');
    nccreate(outputFileName,'y','Dimensions',{'y',ny});
    ncwrite(outputFileName,'y',yCellGrid_urb);
    ncwriteatt(outputFileName,'y','units','m');
    ncwriteatt(outputFileName,'y','long_name','y-distance');
    nccreate(outputFileName,'z','Dimensions',{'z',nz});
    ncwrite(outputFileName,'z',zCellGrid_urb);
    ncwriteatt(outputFileName,'z','units','m');
    ncwriteatt(outputFileName,'z','long_name','z-distance');
    
    terrainInfo = 0;    % for now, use flat terrain, blank slate
    nccreate(outputFileName,'terrain','Dimensions',{'x',nx,'y',ny});
    ncwrite(outputFileName,'terrain',terrainInfo);
    ncwriteatt(outputFileName,'terrain','units','m');
    ncwriteatt(outputFileName,'terrain','long_name','terrain height');
    
    nccreate(outputFileName,'u','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'u',uMean);
    ncwriteatt(outputFileName,'u','units','m s-1');
    ncwriteatt(outputFileName,'u','long_name','x-component velocity');
    nccreate(outputFileName,'v','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'v',vMean);
    ncwriteatt(outputFileName,'v','units','m s-1');
    ncwriteatt(outputFileName,'v','long_name','y-component velocity');
    nccreate(outputFileName,'w','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'w',wMean);
    ncwriteatt(outputFileName,'w','units','m s-1');
    ncwriteatt(outputFileName,'w','long_name','z-component velocity');
    
    icellValues = int32(1); % needs to be int32 datatype. Set it all to 1 for now
    nccreate(outputFileName,'icell','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1},'Datatype','int32');
    ncwrite(outputFileName,'icell',icellValues);
    ncwriteatt(outputFileName,'icell','units','--');
    ncwriteatt(outputFileName,'icell','long_name','icell flag value');
    
    %%% now check to make sure it looks right
    ncdisp(outputFileName);
    
end