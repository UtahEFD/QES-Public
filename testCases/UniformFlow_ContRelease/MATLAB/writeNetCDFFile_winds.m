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

nccreate(outputFileName,'x_face','Dimensions',{'x_face',nx});
nccreate(outputFileName,'y_face','Dimensions',{'y_face',ny});
nccreate(outputFileName,'z_face','Dimensions',{'z_face',nz});


nccreate(outputFileName,'x','Dimensions',{'x',nx_cc});
ncwrite(outputFileName,'x',x_cc);
ncwriteatt(outputFileName,'x','units','m');
ncwriteatt(outputFileName,'x','long_name','x-distance');
nccreate(outputFileName,'y','Dimensions',{'y',ny_cc});
ncwrite(outputFileName,'y',y_cc);
ncwriteatt(outputFileName,'y','units','m');
ncwriteatt(outputFileName,'y','long_name','y-distance');
nccreate(outputFileName,'z','Dimensions',{'z',nz_cc});
ncwrite(outputFileName,'z',z_cc);
ncwriteatt(outputFileName,'z','units','m');
ncwriteatt(outputFileName,'z','long_name','z-distance');

nccreate(outputFileName,'u','Dimensions',{'x_face',nx,'y_face',ny,'z_face',nz,'t',1});
ncwrite(outputFileName,'u',u);
ncwriteatt(outputFileName,'u','units','m s-1');
ncwriteatt(outputFileName,'u','long_name','x-component velocity');
nccreate(outputFileName,'v','Dimensions',{'x_face',nx,'y_face',ny,'z_face',nz,'t',1});
ncwrite(outputFileName,'v',v);
ncwriteatt(outputFileName,'v','units','m s-1');
ncwriteatt(outputFileName,'v','long_name','y-component velocity');
nccreate(outputFileName,'w','Dimensions',{'x_face',nx,'y_face',ny,'z_face',nz,'t',1});
ncwrite(outputFileName,'w',w);
ncwriteatt(outputFileName,'w','units','m s-1');
ncwriteatt(outputFileName,'w','long_name','z-component velocity');

nccreate(outputFileName,'icellflag','Dimensions',{'x',nx_cc,'y',ny_cc,'z',nz_cc,'t',1},'Datatype','int32');
ncwrite(outputFileName,'icellflag',icellflag);
ncwriteatt(outputFileName,'icellflag','units','--');
ncwriteatt(outputFileName,'icellflag','long_name','icell flag value');

%%% now check to make sure it looks right
%ncdisp(outputFileName);

end