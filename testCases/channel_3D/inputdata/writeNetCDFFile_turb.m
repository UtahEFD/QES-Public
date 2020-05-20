function writeNetCDFFile_turb(outputBaseName,x_cc,y_cc,z_cc,CoEps,tke,txx,txy,txz,tyy,tyz,tzz)

% set the output filename
outputFileName = sprintf('%s_turb.nc',outputBaseName);
%%%% if the file already exists, delete it
if exist(outputFileName, 'file') == 2
    delete(outputFileName);
end

% setup some other side variables
nx = length(x_cc);
ny = length(y_cc);
nz = length(z_cc);

times = 0;  % in case need more times later
nccreate(outputFileName,'t','Dimensions',{'t',1},'Format','netcdf4');
ncwrite(outputFileName,'t',times);
ncwriteatt(outputFileName,'t','units','s');
ncwriteatt(outputFileName,'t','long_name','time');
nccreate(outputFileName,'x','Dimensions',{'x',nx});
ncwrite(outputFileName,'x',x_cc);
ncwriteatt(outputFileName,'x','units','m');
ncwriteatt(outputFileName,'x','long_name','x-distance');
nccreate(outputFileName,'y','Dimensions',{'y',ny});
ncwrite(outputFileName,'y',y_cc);
ncwriteatt(outputFileName,'y','units','m');
ncwriteatt(outputFileName,'y','long_name','y-distance');
nccreate(outputFileName,'z','Dimensions',{'z',nz});
ncwrite(outputFileName,'z',z_cc);
ncwriteatt(outputFileName,'z','units','m');
ncwriteatt(outputFileName,'z','long_name','z-distance');

nccreate(outputFileName,'tau_11','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tau_11',txx);
ncwriteatt(outputFileName,'tau_11','units','kg m-1 s-2');
ncwriteatt(outputFileName,'tau_11','long_name','uu stress');
nccreate(outputFileName,'tau_12','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tau_12',txy);
ncwriteatt(outputFileName,'tau_12','units','kg m-1 s-2');
ncwriteatt(outputFileName,'tau_12','long_name','uv stress');
nccreate(outputFileName,'tau_13','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tau_13',txz);
ncwriteatt(outputFileName,'tau_13','units','kg m-1 s-2');
ncwriteatt(outputFileName,'tau_13','long_name','uw stress');
nccreate(outputFileName,'tau_22','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tau_22',tyy);
ncwriteatt(outputFileName,'tau_22','units','kg m-1 s-2');
ncwriteatt(outputFileName,'tau_22','long_name','vv stress');
nccreate(outputFileName,'tau_23','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tau_23',tyz);
ncwriteatt(outputFileName,'tau_23','units','kg m-1 s-2');
ncwriteatt(outputFileName,'tau_23','long_name','vw stress');
nccreate(outputFileName,'tau_33','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tau_33',tzz);
ncwriteatt(outputFileName,'tau_33','units','kg m-1 s-2');
ncwriteatt(outputFileName,'tau_33','long_name','ww stress');



nccreate(outputFileName,'CoEps','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'CoEps',CoEps);
ncwriteatt(outputFileName,'CoEps','units','m2 s-3');
ncwriteatt(outputFileName,'CoEps','long_name','CoEps');

nccreate(outputFileName,'tke','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tke',tke);
ncwriteatt(outputFileName,'tke','units','m2 s-2');
ncwriteatt(outputFileName,'tke','long_name','tke');

%%% now check to make sure it looks right
ncdisp(outputFileName);

end