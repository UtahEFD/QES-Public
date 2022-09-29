function writeNetCDFFile_turb(outputBaseName,x,y,z,CoEps,tke,txx,txy,txz,tyy,tyz,tzz)

% set the output filename
outputFileName = sprintf('%s_turbOut.nc',outputBaseName);
%%%% if the file already exists, delete it
if exist(outputFileName, 'file') == 2
    delete(outputFileName);
end

% setup some other side variables
nx = length(x);
ny = length(y);
nz = length(z);

times = 0;  % in case need more times later

nc = netcdf.create(outputFileName, 'NETCDF4');

dim_t = netcdf.defDim(nc, 't', netcdf.getConstant('NC_UNLIMITED'));

dim_x = netcdf.defDim(nc,'x',nx);
dim_y = netcdf.defDim(nc,'y',ny);
dim_z = netcdf.defDim(nc,'z',nz);

varid_t = netcdf.defVar(nc, 't', 'float', []);
netcdf.putAtt(nc,varid_t,'long_name','time');
netcdf.putAtt(nc,varid_t,'units','s');
netcdf.putAtt(nc,varid_t,'cartesian_axis','T');

varid_x = netcdf.defVar(nc, 'x', 'float', dim_x);
netcdf.putAtt(nc,varid_x,'long_name','x-distance');
netcdf.putAtt(nc,varid_x,'units','m');
netcdf.putAtt(nc,varid_x,'cartesian_axis','X');

varid_y = netcdf.defVar(nc, 'y', 'float', dim_y);
netcdf.putAtt(nc,varid_y,'long_name','y-distance');
netcdf.putAtt(nc,varid_y,'units','m');
netcdf.putAtt(nc,varid_y,'cartesian_axis','Y');

varid_z = netcdf.defVar(nc, 'z', 'float', dim_z);
netcdf.putAtt(nc,varid_z,'long_name','z-distance');
netcdf.putAtt(nc,varid_z,'units','m');
netcdf.putAtt(nc,varid_z,'cartesian_axis','Z');

% -- STRESS ---------------------------------------------------------------
varid_txx = netcdf.defVar(nc, 'txx', 'float', [dim_x dim_y dim_z dim_t]);
netcdf.putAtt(nc,varid_txx,'long_name','txx = uu stress');
netcdf.putAtt(nc,varid_txx,'units','m2 s-2');

varid_txy = netcdf.defVar(nc, 'txy', 'float', [dim_x dim_y dim_z dim_t]);
netcdf.putAtt(nc,varid_txy,'long_name','txy = uv stress');
netcdf.putAtt(nc,varid_txy,'units','m2 s-2');

varid_txz = netcdf.defVar(nc, 'txz', 'float', [dim_x dim_y dim_z dim_t]);
netcdf.putAtt(nc,varid_txz,'long_name','txz = uw stress');
netcdf.putAtt(nc,varid_txz,'units','m2 s-2');

varid_tyy = netcdf.defVar(nc, 'tyy', 'float', [dim_x dim_y dim_z dim_t]);
netcdf.putAtt(nc,varid_tyy,'long_name','tyy = vv stress');
netcdf.putAtt(nc,varid_tyy,'units','m2 s-2');

varid_tyz = netcdf.defVar(nc, 'tyz', 'float', [dim_x dim_y dim_z dim_t]);
netcdf.putAtt(nc,varid_tyz,'long_name','tyz = vw stress');
netcdf.putAtt(nc,varid_tyz,'units','m2 s-2');

varid_tzz = netcdf.defVar(nc, 'tzz', 'float', [dim_x dim_y dim_z dim_t]);
netcdf.putAtt(nc,varid_tzz,'long_name','tzz = ww stress');
netcdf.putAtt(nc,varid_tzz,'units','m2 s-2');

% -- TKE ------------------------------------------------------------------
varid_eps = netcdf.defVar(nc, 'CoEps', 'float', [dim_x dim_y dim_z dim_t]);
netcdf.putAtt(nc,varid_eps,'long_name','dissipation rate of tke');
netcdf.putAtt(nc,varid_eps,'units','m2 s-3');

varid_tke = netcdf.defVar(nc, 'tke', 'float', [dim_x dim_y dim_z dim_t]);
netcdf.putAtt(nc,varid_tke,'long_name','tke = turbulence kinetic energy');
netcdf.putAtt(nc,varid_tke,'units','m2 s-2');

netcdf.endDef(nc);

netcdf.putVar(nc,varid_t, times);
netcdf.putVar(nc,varid_x, x);
netcdf.putVar(nc,varid_y, y);
netcdf.putVar(nc,varid_z, z);

netcdf.putVar(nc, varid_txx, [0 0 0 0], [nx ny nz 1], txx);
netcdf.putVar(nc, varid_txy, [0 0 0 0], [nx ny nz 1], txy);
netcdf.putVar(nc, varid_txz, [0 0 0 0], [nx ny nz 1], txz);
netcdf.putVar(nc, varid_tyy, [0 0 0 0], [nx ny nz 1], tyy);
netcdf.putVar(nc, varid_tyz, [0 0 0 0], [nx ny nz 1], tyz);
netcdf.putVar(nc, varid_tzz, [0 0 0 0], [nx ny nz 1], tzz);

netcdf.putVar(nc, varid_eps, [0 0 0 0], [nx ny nz 1], CoEps);
netcdf.putVar(nc, varid_tke, [0 0 0 0], [nx ny nz 1], tke);

netcdf.close(nc)

% nccreate(outputFileName,'t','Dimensions',{'t',1},'Format','netcdf4');
% ncwrite(outputFileName,'t',times);
% ncwriteatt(outputFileName,'t','units','s');
% ncwriteatt(outputFileName,'t','long_name','time');
% nccreate(outputFileName,'x','Dimensions',{'x',nx});
% ncwrite(outputFileName,'x',x);
% ncwriteatt(outputFileName,'x','units','m');
% ncwriteatt(outputFileName,'x','long_name','x-distance');
% nccreate(outputFileName,'y','Dimensions',{'y',ny});
% ncwrite(outputFileName,'y',y);
% ncwriteatt(outputFileName,'y','units','m');
% ncwriteatt(outputFileName,'y','long_name','y-distance');
% nccreate(outputFileName,'z','Dimensions',{'z',nz});
% ncwrite(outputFileName,'z',z);
% ncwriteatt(outputFileName,'z','units','m');
% ncwriteatt(outputFileName,'z','long_name','z-distance');
% 
% nccreate(outputFileName,'txx','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
% ncwrite(outputFileName,'txx',txx);
% ncwriteatt(outputFileName,'txx','units','kg m-1 s-2');
% ncwriteatt(outputFileName,'txx','long_name','uu stress');
% nccreate(outputFileName,'txy','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
% ncwrite(outputFileName,'txy',txy);
% ncwriteatt(outputFileName,'txy','units','kg m-1 s-2');
% ncwriteatt(outputFileName,'txy','long_name','uv stress');
% nccreate(outputFileName,'txz','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
% ncwrite(outputFileName,'txz',txz);
% ncwriteatt(outputFileName,'txz','units','kg m-1 s-2');
% ncwriteatt(outputFileName,'txz','long_name','uw stress');
% nccreate(outputFileName,'tyy','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
% ncwrite(outputFileName,'tyy',tyy);
% ncwriteatt(outputFileName,'tyy','units','kg m-1 s-2');
% ncwriteatt(outputFileName,'tyy','long_name','vv stress');
% nccreate(outputFileName,'tyz','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
% ncwrite(outputFileName,'tyz',tyz);
% ncwriteatt(outputFileName,'tyz','units','kg m-1 s-2');
% ncwriteatt(outputFileName,'tyz','long_name','vw stress');
% nccreate(outputFileName,'tzz','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
% ncwrite(outputFileName,'tzz',tzz);
% ncwriteatt(outputFileName,'tzz','units','kg m-1 s-2');
% ncwriteatt(outputFileName,'tzz','long_name','ww stress');
% 
% nccreate(outputFileName,'CoEps','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
% ncwrite(outputFileName,'CoEps',CoEps);
% ncwriteatt(outputFileName,'CoEps','units','m2 s-3');
% ncwriteatt(outputFileName,'CoEps','long_name','CoEps');
% 
% nccreate(outputFileName,'tke','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
% ncwrite(outputFileName,'tke',tke);
% ncwriteatt(outputFileName,'tke','units','m2 s-2');
% ncwriteatt(outputFileName,'tke','long_name','tke');

%%% now check to make sure it looks right
%ncdisp(outputFileName);

end