function [simVarsExist,  urb_t,urb_x,urb_y,urb_z, urb_u,urb_v,urb_w,turb_sig_x,turb_sig_y,turb_sig_z,turb_txx,turb_txy,turb_txz,turb_tyy,turb_tyz,turb_tzz,turb_epps,turb_tke,eul_dtxxdx,eul_dtxydy,eul_dtxzdz,eul_dtxydx,eul_dtyydy,eul_dtyzdz,eul_dtxzdx,eul_dtyzdy,eul_dtzzdz,eul_flux_div_x,eul_flux_div_y,eul_flux_div_z] = readNetcdfEulerianFile(eulFile)

    % not sure of another way to catch if the netcdf read stuff fails, so
    % going to assume any error means that simVars do NOT exist
    try
        
        %%% if desired, get info about the urb file
% %         ncdisp(eulFile);

        %%% ncdisp shows variables t, x, y, z,  u, v, w, sig_x, sig_y, sig_z
        %%%  txx, txy, txz, tyy, tyz, tzz, epps, tke, dtxxdx, dtxydy, dtxzdz,
        %%%  dtxydx, dtyydy, dtyzdz, dtxzdx, dtyzdy, dtzzdz, flux_div_x,
        %%%  flux_div_y, flux_div_z
        %%% where t is size 1x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
        %%% where x is size 200x1 in dimensions (x) with units 'm' and long_name 'x-distance' with datatype double,
        %%% where y is size 200x1 in dimensions (y) with units 'm' and long_name 'y-distance' with datatype double,
        %%% where z is size 200x1 in dimensions (z) with units 'm' and long_name 'z-distance' with datatype double,
        %%% where u is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'x-component mean velocity' with datatype double,
        %%% where v is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'y-component mean velocity' with datatype double,
        %%% where w is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'z-component mean velocity' with datatype double,
        %%% where sig_x is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'x-component variance' with datatype double,
        %%% where sig_y is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'y-component variance' with datatype double,
        %%% where sig_z is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'z-component variance' with datatype double,
        %%% where txx is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'uu-component of stress tensor' with datatype double,
        %%% where txy is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'uv-component of stress tensor' with datatype double,
        %%% where txz is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'uw-component of stress tensor' with datatype double,
        %%% where tyy is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'vv-component of stress tensor' with datatype double,
        %%% where tyz is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'vw-component of stress tensor' with datatype double,
        %%% where tzz is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'ww-component of stress tensor' with datatype double,
        %%% where epps is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-3' and long_name dissipation rate' with datatype single,
        %%% where tke is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'turbulent kinetic energy' with datatype double,
        %%% where dtxxdx is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'derivative of txx in the x direction' with datatype double,
        %%% where dtxydy is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'derivative of txy in the y direction' with datatype double,
        %%% where dtxzdz is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'derivative of txz in the z direction' with datatype double,
        %%% where dtxydx is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'derivative of txy in the x direction' with datatype double,
        %%% where dtyydy is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'derivative of tyy in the y direction' with datatype double,
        %%% where dtyzdz is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'derivative of tyz in the z direction' with datatype double,
        %%% where dtxzdx is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'derivative of txz in the x direction' with datatype double,
        %%% where dtyzdy is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'derivative of tyz in the y direction' with datatype double,
        %%% where dtzzdz is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'derivative of tzz in the z direction' with datatype double,
        %%% where flux_div_x is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'momentum flux through the x-plane' with datatype double,
        %%% where flux_div_y is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'momentum flux through the y-plane' with datatype double,
        %%% where flux_div_z is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-2' and long_name 'momentum flux through the z-plane' with datatype double.
        %%% t is zero. x,y, and z are just linspace from domainStart to domainEnd

        urb_t = ncread(eulFile,'t');
        urb_x = ncread(eulFile,'x');
        urb_y = ncread(eulFile,'y');
        urb_z = ncread(eulFile,'z');
        urb_u = ncread(eulFile,'u');
        urb_v = ncread(eulFile,'v');
        urb_w = ncread(eulFile,'w');
        turb_sig_x = ncread(eulFile,'sig_x');
        turb_sig_y = ncread(eulFile,'sig_y');
        turb_sig_z = ncread(eulFile,'sig_z');
        turb_txx = ncread(eulFile,'txx');
        turb_txy = ncread(eulFile,'txy');
        turb_txz = ncread(eulFile,'txz');
        turb_tyy = ncread(eulFile,'tyy');
        turb_tyz = ncread(eulFile,'tyz');
        turb_tzz = ncread(eulFile,'tzz');
        turb_epps = ncread(eulFile,'epps');
        turb_tke = ncread(eulFile,'tke');
        eul_dtxxdx = ncread(eulFile,'dtxxdx');
        eul_dtxydy = ncread(eulFile,'dtxydy');
        eul_dtxzdz = ncread(eulFile,'dtxzdz');
        eul_dtxydx = ncread(eulFile,'dtxydx');
        eul_dtyydy = ncread(eulFile,'dtyydy');
        eul_dtyzdz = ncread(eulFile,'dtyzdz');
        eul_dtxzdx = ncread(eulFile,'dtxzdx');
        eul_dtyzdy = ncread(eulFile,'dtyzdy');
        eul_dtzzdz = ncread(eulFile,'dtzzdz');
        eul_flux_div_x = ncread(eulFile,'flux_div_x');
        eul_flux_div_y = ncread(eulFile,'flux_div_y');
        eul_flux_div_z = ncread(eulFile,'flux_div_z');
        
        % got to here without failing the try catch statement, so simVars
        % DO exist
        simVarsExist = true;
        
    catch
        
        simVarsExist = false;
        
        % possible nothing was loaded, so need to set all the output to NAN
        urb_t = NaN;
        urb_x = NaN;
        urb_y = NaN;
        urb_z = NaN;
        urb_u = NaN;
        urb_v = NaN;
        urb_w = NaN;
        turb_sig_x = NaN;
        turb_sig_y = NaN;
        turb_sig_z = NaN;
        turb_txx = NaN;
        turb_txy = NaN;
        turb_txz = NaN;
        turb_tyy = NaN;
        turb_tyz = NaN;
        turb_tzz = NaN;
        turb_epps = NaN;
        turb_tke = NaN;
        eul_dtxxdx = NaN;
        eul_dtxydy = NaN;
        eul_dtxzdz = NaN;
        eul_dtxydx = NaN;
        eul_dtyydy = NaN;
        eul_dtyzdz = NaN;
        eul_dtxzdx = NaN;
        eul_dtyzdy = NaN;
        eul_dtzzdz = NaN;
        eul_flux_div_x = NaN;
        eul_flux_div_y = NaN;
        eul_flux_div_z = NaN;
        
    end

end