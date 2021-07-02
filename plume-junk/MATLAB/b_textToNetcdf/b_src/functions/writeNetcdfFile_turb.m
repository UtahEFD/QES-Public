function writeNetcdfFile_turb(outputDir,outputBaseName,  xCellGrid_turb,yCellGrid_turb,zCellGrid_turb, C0,epps,sigma2,txx,txy,txz,tyy,tyz,tzz)

    % verify the outputDir is a valid path
    if exist(outputDir, 'dir') ~= 7
        error('!!! writeNetcdfOutput_turb error !!! input outputDir \"%s\" does not exist or is not a valid directory!',outputDir);
    end
    
    % set the output filename
    outputFileName = sprintf('%s/%s_turb.nc',outputDir,outputBaseName);
    
    %%%% if the file already exists, delete it
    if exist(outputFileName, 'file') == 2
        delete(outputFileName);
    end
    
    % setup some other side variables
    nx = length(xCellGrid_turb);
    ny = length(yCellGrid_turb);
    nz = length(zCellGrid_turb);
    
    % don't worry if the output already exists, just overwrite it
    %%%save(outputFileName,  'xCellGrid_turb','yCellGrid_turb','zCellGrid_turb', 'epps','sigma2','txx','txy','txz','tyy','tyz','tzz');
    times = 0;  % in case need more times later
    % for some odd reason is unlimited format in urb, but not here, so not adding Inf to the dimension variable in turb though it is done that way in urb
    nccreate(outputFileName,'t','Dimensions',{'t',1},'Format','netcdf4');
    ncwrite(outputFileName,'t',times);
    ncwriteatt(outputFileName,'t','units','s');
    ncwriteatt(outputFileName,'t','long_name','time');
    nccreate(outputFileName,'x','Dimensions',{'x',nx});
    ncwrite(outputFileName,'x',xCellGrid_turb);
    ncwriteatt(outputFileName,'x','units','m');
    ncwriteatt(outputFileName,'x','long_name','x-distance');
    nccreate(outputFileName,'y','Dimensions',{'y',ny});
    ncwrite(outputFileName,'y',yCellGrid_turb);
    ncwriteatt(outputFileName,'y','units','m');
    ncwriteatt(outputFileName,'y','long_name','y-distance');
    nccreate(outputFileName,'z','Dimensions',{'z',nz});
    ncwrite(outputFileName,'z',zCellGrid_turb);
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
    
    nccreate(outputFileName,'sig_11','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'sig_11',sigma2);
    ncwriteatt(outputFileName,'sig_11','units','m s-1');
    ncwriteatt(outputFileName,'sig_11','long_name','uu sigma');
    nccreate(outputFileName,'sig_22','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'sig_22',sigma2);
    ncwriteatt(outputFileName,'sig_22','units','m s-1');
    ncwriteatt(outputFileName,'sig_22','long_name','vv sigma');
    nccreate(outputFileName,'sig_33','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'sig_33',sigma2);
    ncwriteatt(outputFileName,'sig_33','units','m s-1');
    ncwriteatt(outputFileName,'sig_33','long_name','ww sigma');
    
    lambdaVal = 0;  % this is not used, so going to fill it with crap
    nccreate(outputFileName,'lam_11','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'lam_11',lambdaVal);
    ncwriteatt(outputFileName,'lam_11','units','m s2 kg-1');
    ncwriteatt(outputFileName,'lam_11','long_name','uu lambda');
    nccreate(outputFileName,'lam_12','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'lam_12',lambdaVal);
    ncwriteatt(outputFileName,'lam_12','units','m s2 kg-1');
    ncwriteatt(outputFileName,'lam_12','long_name','uv lambda');
    nccreate(outputFileName,'lam_13','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'lam_13',lambdaVal);
    ncwriteatt(outputFileName,'lam_13','units','m s2 kg-1');
    ncwriteatt(outputFileName,'lam_13','long_name','uw lambda');
    nccreate(outputFileName,'lam_21','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'lam_21',lambdaVal);
    ncwriteatt(outputFileName,'lam_21','units','m s2 kg-1');
    ncwriteatt(outputFileName,'lam_21','long_name','vu lambda');
    nccreate(outputFileName,'lam_22','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'lam_22',lambdaVal);
    ncwriteatt(outputFileName,'lam_22','units','m s2 kg-1');
    ncwriteatt(outputFileName,'lam_22','long_name','vv lambda');
    nccreate(outputFileName,'lam_23','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'lam_23',lambdaVal);
    ncwriteatt(outputFileName,'lam_23','units','m s2 kg-1');
    ncwriteatt(outputFileName,'lam_23','long_name','vw lambda');
    nccreate(outputFileName,'lam_31','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'lam_31',lambdaVal);
    ncwriteatt(outputFileName,'lam_31','units','m s2 kg-1');
    ncwriteatt(outputFileName,'lam_31','long_name','wu lambda');
    nccreate(outputFileName,'lam_32','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'lam_32',lambdaVal);
    ncwriteatt(outputFileName,'lam_32','units','m s2 kg-1');
    ncwriteatt(outputFileName,'lam_32','long_name','wv lambda');
    nccreate(outputFileName,'lam_33','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'lam_33',lambdaVal);
    ncwriteatt(outputFileName,'lam_33','units','m s2 kg-1');
    ncwriteatt(outputFileName,'lam_33','long_name','ww lambda');
    
    velDerivVals = 0;   % this is not used, so going to fill it with crap
    nccreate(outputFileName,'dudx','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'dudx',velDerivVals);
    ncwriteatt(outputFileName,'dudx','units','s-1');
    ncwriteatt(outputFileName,'dudx','long_name','dudx');
    nccreate(outputFileName,'dvdx','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'dvdx',velDerivVals);
    ncwriteatt(outputFileName,'dvdx','units','s-1');
    ncwriteatt(outputFileName,'dvdx','long_name','dvdx');
    nccreate(outputFileName,'dwdx','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'dwdx',velDerivVals);
    ncwriteatt(outputFileName,'dwdx','units','s-1');
    ncwriteatt(outputFileName,'dwdx','long_name','dwdx');
    nccreate(outputFileName,'dudy','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'dudy',velDerivVals);
    ncwriteatt(outputFileName,'dudy','units','s-1');
    ncwriteatt(outputFileName,'dudy','long_name','dudy');
    nccreate(outputFileName,'dvdy','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'dvdy',velDerivVals);
    ncwriteatt(outputFileName,'dvdy','units','s-1');
    ncwriteatt(outputFileName,'dvdy','long_name','dvdy');
    nccreate(outputFileName,'dwdy','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'dwdy',velDerivVals);
    ncwriteatt(outputFileName,'dwdy','units','s-1');
    ncwriteatt(outputFileName,'dwdy','long_name','dwdy');
    nccreate(outputFileName,'dudz','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'dudz',velDerivVals);
    ncwriteatt(outputFileName,'dudz','units','s-1');
    ncwriteatt(outputFileName,'dudz','long_name','dudz');
    nccreate(outputFileName,'dvdz','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'dvdz',velDerivVals);
    ncwriteatt(outputFileName,'dvdz','units','s-1');
    ncwriteatt(outputFileName,'dvdz','long_name','dvdz');
    nccreate(outputFileName,'dwdz','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'dwdz',velDerivVals);
    ncwriteatt(outputFileName,'dwdz','units','s-1');
    ncwriteatt(outputFileName,'dwdz','long_name','dwdz');
    
    CoEpsValues = C0*epps;
    nccreate(outputFileName,'CoEps','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'CoEps',CoEpsValues);
    ncwriteatt(outputFileName,'CoEps','units','m2 s-3');
    ncwriteatt(outputFileName,'CoEps','long_name','CoEps');
    
    tkeValues = 0;   % this is not used, so going to fill it with crap
    nccreate(outputFileName,'tke','Dimensions',{'x',nx,'y',ny,'z',nz,'t',1});
    ncwrite(outputFileName,'tke',tkeValues);
    ncwriteatt(outputFileName,'tke','units','m2 s-2');
    ncwriteatt(outputFileName,'tke','long_name','tke');
    
    %%% now check to make sure it looks right
    ncdisp(outputFileName);
    
end