function writeNetCDFFile_winds(outputBaseName,nx,ny,nz,x_cc,y_cc,z_cc,u,v,w,icellflag)
    
% set the output filename
outputFileName = sprintf('%s_windsWk.nc',outputBaseName);
%%%% if the file already exists, delete it
if exist(outputFileName, 'file') == 2
    delete(outputFileName);
end


% setup some other side variables
nx_cc = length(x_cc);
ny_cc = length(y_cc);
nz_cc = length(z_cc);

% don't worry if the output already exists, just overwrite it
%%%save(outputFileName,  'xCellGrid_urb','yCellGrid_urb','zCellGrid_urb', 'uMean','vMean','wMean');
times = 0;  % in case need more times later
nccreate(outputFileName,'t','Dimensions',{'t',1,Inf},'Format','netcdf4');
ncwrite(outputFileName,'t',times);
ncwriteatt(outputFileName,'t','units','s');
ncwriteatt(outputFileName,'t','long_name','time');

nccreate(outputFileName,'x','Dimensions',{'x',nx});
nccreate(outputFileName,'y','Dimensions',{'y',ny});
nccreate(outputFileName,'z','Dimensions',{'z',nz});

nccreate(outputFileName,'x_cc','Dimensions',{'x_cc',nx_cc});
ncwrite(outputFileName,'x_cc',x_cc);
ncwriteatt(outputFileName,'x_cc','units','m');
ncwriteatt(outputFileName,'x_cc','long_name','x-distance');
nccreate(outputFileName,'y_cc','Dimensions',{'y_cc',ny_cc});
ncwrite(outputFileName,'y_cc',y_cc);
ncwriteatt(outputFileName,'y_cc','units','m');
ncwriteatt(outputFileName,'y_cc','long_name','y-distance');
nccreate(outputFileName,'z_cc','Dimensions',{'z_cc',nz_cc});
ncwrite(outputFileName,'z_cc',z_cc);
ncwriteatt(outputFileName,'z_cc','units','m');
ncwriteatt(outputFileName,'z_cc','long_name','z-distance');

nccreate(outputFileName,'u','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'u',u);
ncwriteatt(outputFileName,'u','units','m s-1');
ncwriteatt(outputFileName,'u','long_name','x-component velocity');
nccreate(outputFileName,'v','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'v',v);
ncwriteatt(outputFileName,'v','units','m s-1');
ncwriteatt(outputFileName,'v','long_name','y-component velocity');
nccreate(outputFileName,'w','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'w',w);
ncwriteatt(outputFileName,'w','units','m s-1');
ncwriteatt(outputFileName,'w','long_name','z-component velocity');

nccreate(outputFileName,'icellflag','Dimensions',{'x_cc',nx_cc,'y_cc',ny_cc,'z_cc',nz_cc,'t',1},'Datatype','int32');
ncwrite(outputFileName,'icellflag',icellflag);
ncwriteatt(outputFileName,'icellflag','units','--');
ncwriteatt(outputFileName,'icellflag','long_name','icell flag value');

%%% now check to make sure it looks right
ncdisp(outputFileName);

end