function writeNetCDFFile_winds(outputBaseName,nx,ny,nz,x_cc,y_cc,z_cc,u,v,w,icellflag)

% set the output filename
outputFileName = sprintf('%s_windsWk.nc',outputBaseName);
%%%% if the file already exists, delete it
if exist(outputFileName, 'file') == 2
    delete(outputFileName);
end
if exist('testfile.nc', 'file') == 2
    delete('testfile.nc');
end

% setup some other side variables
nx_cc = length(x_cc);
ny_cc = length(y_cc);
nz_cc = length(z_cc);

z_face = zeros(nz,1);
z_face(2:end) = (0:(nz_cc-1))*(z_cc(2)-z_cc(1));
z_face(1) = - z_face(3);

% don't worry if the output already exists, just overwrite it
%%%save(outputFileName,  'xCellGrid_urb','yCellGrid_urb','zCellGrid_urb', 'uMean','vMean','wMean');
times = 0;  % in case need more times later

nc = netcdf.create(outputFileName, 'NETCDF4');

dim_t = netcdf.defDim(nc, 't', netcdf.getConstant('NC_UNLIMITED'));

dim_x_face = netcdf.defDim(nc,'x_face',nx);
dim_y_face = netcdf.defDim(nc,'y_face',ny);
dim_z_face = netcdf.defDim(nc,'z_face',nz);

dim_x_cell = netcdf.defDim(nc,'x',nx_cc);
dim_y_cell = netcdf.defDim(nc,'y',ny_cc);
dim_z_cell = netcdf.defDim(nc,'z',nz_cc);

varid_t = netcdf.defVar(nc, 't', 'float', []);
netcdf.putAtt(nc,varid_t,'long_name','time');
netcdf.putAtt(nc,varid_t,'units','s');
netcdf.putAtt(nc,varid_t,'cartesian_axis','T');

varid_x_cell = netcdf.defVar(nc, 'x', 'float', dim_x_cell);
netcdf.putAtt(nc,varid_x_cell,'long_name','x-distance');
netcdf.putAtt(nc,varid_x_cell,'units','m');
netcdf.putAtt(nc,varid_x_cell,'cartesian_axis','X');

varid_y_cell = netcdf.defVar(nc, 'y', 'float', dim_y_cell);
netcdf.putAtt(nc,varid_y_cell,'long_name','y-distance');
netcdf.putAtt(nc,varid_y_cell,'units','m');
netcdf.putAtt(nc,varid_y_cell,'cartesian_axis','Y');

varid_z_cell = netcdf.defVar(nc, 'z', 'float', dim_z_cell);
netcdf.putAtt(nc,varid_z_cell,'long_name','z-distance');
netcdf.putAtt(nc,varid_z_cell,'units','m');
netcdf.putAtt(nc,varid_z_cell,'cartesian_axis','Z');

varid_z_face = netcdf.defVar(nc, 'z_face', 'float', dim_z_face);
netcdf.putAtt(nc,varid_z_face,'long_name','z-distance');
netcdf.putAtt(nc,varid_z_face,'units','m');

varid_u = netcdf.defVar(nc, 'u', 'float', [dim_x_face dim_y_face dim_z_face dim_t]);
netcdf.putAtt(nc,varid_u,'long_name','x-component velocity');
netcdf.putAtt(nc,varid_u,'units','m s-1');

varid_v = netcdf.defVar(nc, 'v', 'float', [dim_x_face dim_y_face dim_z_face dim_t]);
netcdf.putAtt(nc,varid_v,'long_name','y-component velocity');
netcdf.putAtt(nc,varid_v,'units','m s-1');

varid_w = netcdf.defVar(nc, 'w', 'float', [dim_x_face dim_y_face dim_z_face dim_t]);
netcdf.putAtt(nc,varid_w,'long_name','z-component velocity');
netcdf.putAtt(nc,varid_w,'units','m s-1');

varid_c = netcdf.defVar(nc, 'icellflag', 'int', [dim_x_cell dim_y_cell dim_z_cell dim_t]);
netcdf.putAtt(nc,varid_c,'long_name','icell flag value');
netcdf.putAtt(nc,varid_c,'units','--');

netcdf.endDef(nc);

netcdf.putVar(nc,varid_t, times);
netcdf.putVar(nc,varid_x_cell, x_cc);
netcdf.putVar(nc,varid_y_cell, y_cc);
netcdf.putVar(nc,varid_z_cell, z_cc);
netcdf.putVar(nc,varid_z_face, z_face);

netcdf.putVar(nc, varid_u, [0 0 0 0], [nx ny nz 1], u);
netcdf.putVar(nc, varid_v, [0 0 0 0], [nx ny nz 1], v);
netcdf.putVar(nc, varid_w, [0 0 0 0], [nx ny nz 1], w);
netcdf.putVar(nc, varid_c, [0 0 0 0], [nx_cc ny_cc nz_cc 1], icellflag);

netcdf.close(nc)


% nccreate(outputFileName,'t','Dimensions',{'t',1,Inf},'Format','netcdf4');
% ncwrite(outputFileName,'t',times);
% ncwriteatt(outputFileName,'t','units','s');
% ncwriteatt(outputFileName,'t','long_name','time');
%
% nccreate(outputFileName,'x_face','Dimensions',{'x_face',nx});
% nccreate(outputFileName,'y_face','Dimensions',{'y_face',ny});
% nccreate(outputFileName,'z_face','Dimensions',{'z_face',nz});
%
%
% nccreate(outputFileName,'x','Dimensions',{'x',nx_cc});
% ncwrite(outputFileName,'x',x_cc);
% ncwriteatt(outputFileName,'x','units','m');
% ncwriteatt(outputFileName,'x','long_name','x-distance');
% nccreate(outputFileName,'y','Dimensions',{'y',ny_cc});
% ncwrite(outputFileName,'y',y_cc);
% ncwriteatt(outputFileName,'y','units','m');
% ncwriteatt(outputFileName,'y','long_name','y-distance');
% nccreate(outputFileName,'z','Dimensions',{'z',nz_cc});
% ncwrite(outputFileName,'z',z_cc);
% ncwriteatt(outputFileName,'z','units','m');
% ncwriteatt(outputFileName,'z','long_name','z-distance');
%
% nccreate(outputFileName,'u','Dimensions',{'x_face',nx,'y_face',ny,'z_face',nz,'t',1});
% ncwrite(outputFileName,'u',u);
% ncwriteatt(outputFileName,'u','units','m s-1');
% ncwriteatt(outputFileName,'u','long_name','x-component velocity');
% nccreate(outputFileName,'v','Dimensions',{'x_face',nx,'y_face',ny,'z_face',nz,'t',1});
% ncwrite(outputFileName,'v',v);
% ncwriteatt(outputFileName,'v','units','m s-1');
% ncwriteatt(outputFileName,'v','long_name','y-component velocity');
% nccreate(outputFileName,'w','Dimensions',{'x_face',nx,'y_face',ny,'z_face',nz,'t',1});
% ncwrite(outputFileName,'w',w);
% ncwriteatt(outputFileName,'w','units','m s-1');
% ncwriteatt(outputFileName,'w','long_name','z-component velocity');
%
% nccreate(outputFileName,'icellflag','Dimensions',{'x',nx_cc,'y',ny_cc,'z',nz_cc,'t',1},'Datatype','int32');
% ncwrite(outputFileName,'icellflag',icellflag);
% ncwriteatt(outputFileName,'icellflag','units','--');
% ncwriteatt(outputFileName,'icellflag','long_name','icell flag value');

%%% now check to make sure it looks right
%ncdisp(outputFileName);

end