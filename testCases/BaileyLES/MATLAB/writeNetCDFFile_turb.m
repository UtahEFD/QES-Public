function writeNetCDFFile_turb(outputBaseName,x_cc,y_cc,z_cc,CoEps,tke,txx,txy,txz,tyy,tyz,tzz)

% set the output filename
outputFileName = sprintf('%s_turbOut.nc',outputBaseName);
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

nccreate(outputFileName,'txx','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'txx',txx);
ncwriteatt(outputFileName,'txx','units','kg m-1 s-2');
ncwriteatt(outputFileName,'txx','long_name','uu stress');
nccreate(outputFileName,'txy','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'txy',txy);
ncwriteatt(outputFileName,'txy','units','kg m-1 s-2');
ncwriteatt(outputFileName,'txy','long_name','uv stress');
nccreate(outputFileName,'txz','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'txz',txz);
ncwriteatt(outputFileName,'txz','units','kg m-1 s-2');
ncwriteatt(outputFileName,'txz','long_name','uw stress');
nccreate(outputFileName,'tyy','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tyy',tyy);
ncwriteatt(outputFileName,'tyy','units','kg m-1 s-2');
ncwriteatt(outputFileName,'tyy','long_name','vv stress');
nccreate(outputFileName,'tyz','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tyz',tyz);
ncwriteatt(outputFileName,'tyz','units','kg m-1 s-2');
ncwriteatt(outputFileName,'tyz','long_name','vw stress');
nccreate(outputFileName,'tzz','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tzz',tzz);
ncwriteatt(outputFileName,'tzz','units','kg m-1 s-2');
ncwriteatt(outputFileName,'tzz','long_name','ww stress');

nccreate(outputFileName,'CoEps','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'CoEps',CoEps);
ncwriteatt(outputFileName,'CoEps','units','m2 s-3');
ncwriteatt(outputFileName,'CoEps','long_name','CoEps');

nccreate(outputFileName,'tke','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
ncwrite(outputFileName,'tke',tke);
ncwriteatt(outputFileName,'tke','units','m2 s-2');
ncwriteatt(outputFileName,'tke','long_name','tke');

%%% now check to make sure it looks right
%ncdisp(outputFileName);

end