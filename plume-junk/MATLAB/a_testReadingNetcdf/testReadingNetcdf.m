

%%%%%%% standard main function stuff


% clear old stuff
clc;
clear;
close all;
path(pathdef);  % resets the path

% return to current directory
mfile_name          = mfilename('fullpath');
[pathstr,name,ext]  = fileparts(mfile_name);
cd(pathstr);


%%%%%%% end standard main function stuff


%% set overall netcdf file locations and variable names

% the files should be held in the directory of this script
workingDir = pwd;


netcdfFile_urb_FlatTerrain = sprintf('%s/FlatTerrain_urb.nc',workingDir);
netcdfFile_urb_vis_FlatTerrain = sprintf('%s/FlatTerrain_urb_vis.nc',workingDir);
netcdfFile_turb_FlatTerrain = sprintf('%s/FlatTerrain_turb.nc',workingDir);
netcdfFile_conc_FlatTerrain = sprintf('%s/FlatTerrain_singlePoint_conc.nc',workingDir);
netcdfFile_eulerian_FlatTerrain = sprintf('%s/FlatTerrain_singlePoint_eulerianData.nc',workingDir);
netcdfFile_particle_FlatTerrain = sprintf('%s/FlatTerrain_singlePoint_particleInfo.nc',workingDir);


%% read urb netcdf file

% now get info about the urb file
ncdisp(netcdfFile_urb_FlatTerrain);

%%% ncdisp shows variables t, x, y, z, x_cc, y_cc, z_cc,  terrain, icell, u, v, w, 
%%%  e, f, g, h, m, n
%%% where t is size 1x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
%%% where x_cc is size 200x1 in dimensions (x_cc) with units 'm' and long_name 'x-distance' with datatype single,
%%% where y_cc is size 200x1 in dimensions (y_cc) with units 'm' and long_name 'y-distance' with datatype single,
%%% where z_cc is size 201x1 in dimensions (z_cc) with units 'm' and long_name 'z-distance' with datatype single,
%%% where terrain is size 200x200 in dimensions (x_cc,y_cc) with units 'm' and long_name 'terrain height' with datatype single,
%%% where icell is size 200x200x201x1 in dimensions (x_cc,y_cc,z_cc,t) with units '--' and long_name 'icell flag value' with datatype int32,
%%% where u is size 201x201x202x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'x-component velocity' with datatype single,
%%% where v is size 201x201x202x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'y-component velocity' with datatype single,
%%% where w is size 201x201x202x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'z-component velocity' with datatype single,
%%% where e is size 200x200x201x1 in dimensions (x_cc,y_cc,z_cc) with units '--' and long_name 'e cut-cell coefficient' with datatype single,
%%% where f is size 200x200x201x1 in dimensions (x_cc,y_cc,z_cc) with units '--' and long_name 'f cut-cell coefficient' with datatype single,
%%% where g is size 200x200x201x1 in dimensions (x_cc,y_cc,z_cc) with units '--' and long_name 'g cut-cell coefficient' with datatype single,
%%% where h is size 200x200x201x1 in dimensions (x_cc,y_cc,z_cc) with units '--' and long_name 'h cut-cell coefficient' with datatype single,
%%% where m is size 200x200x201x1 in dimensions (x_cc,y_cc,z_cc) with units '--' and long_name 'm cut-cell coefficient' with datatype single,
%%% where n is size 200x200x201x1 in dimensions (x_cc,y_cc,z_cc) with units '--' and long_name 'n cut-cell coefficient' with datatype single.
%%% note that the terrain is full of straight zeros and the icell is full
%%% of only ones except for the first layer which is full of twos.
%%% t is just a zero. x and y are linspaces from domainStart+dx/2 to domainEnd-dx/2 spaced by the corresponding dx
%%% while z is a linspace from domainStart-dx*3/2 to domainEnd-dx/2 spaced by the corresponding dx

urb_t = ncread(netcdfFile_urb_FlatTerrain,'t');
%%% these next three are NOT found in the file, even though they
%%% technically are? Weird
%urb_x = ncread(netcdfFile_urb_FlatTerrain,'x');
%urb_y = ncread(netcdfFile_urb_FlatTerrain,'y');
%urb_z = ncread(netcdfFile_urb_FlatTerrain,'z');
urb_x_cc = ncread(netcdfFile_urb_FlatTerrain,'x_cc');
urb_y_cc = ncread(netcdfFile_urb_FlatTerrain,'y_cc');
urb_z_cc = ncread(netcdfFile_urb_FlatTerrain,'z_cc');
urb_terrain = ncread(netcdfFile_urb_FlatTerrain,'terrain');
urb_icell = ncread(netcdfFile_urb_FlatTerrain,'icell');
urb_u = ncread(netcdfFile_urb_FlatTerrain,'u');
urb_u_units = ncreadatt(netcdfFile_urb_FlatTerrain,'u','units');
urb_u_long_name = ncreadatt(netcdfFile_urb_FlatTerrain,'u','long_name');
urb_v = ncread(netcdfFile_urb_FlatTerrain,'v');
urb_w = ncread(netcdfFile_urb_FlatTerrain,'w');
urb_e = ncread(netcdfFile_urb_FlatTerrain,'e');
urb_f = ncread(netcdfFile_urb_FlatTerrain,'f');
urb_g = ncread(netcdfFile_urb_FlatTerrain,'g');
urb_h = ncread(netcdfFile_urb_FlatTerrain,'h');
urb_m = ncread(netcdfFile_urb_FlatTerrain,'m');
urb_n = ncread(netcdfFile_urb_FlatTerrain,'n');


%% read urb vis netcdf file

% now get info about the urb file
ncdisp(netcdfFile_urb_vis_FlatTerrain);

%%% ncdisp shows variables t, x, y, z, icell, u, v, w
%%% where t is size 1x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
%%% where x is size 200x1 in dimensions (x) with units 'm' and long_name 'x-distance' with datatype single,
%%% where y is size 200x1 in dimensions (y) with units 'm' and long_name 'y-distance' with datatype single,
%%% where z is size 200x1 in dimensions (z) with units 'm' and long_name 'z-distance' with datatype single,
%%% where icell is size 200x200x200x1 in dimensions (x,y,z,t) with units '--' and long_name 'icell flag value' with datatype int32,
%%% where u is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'x-component velocity' with datatype double,
%%% where v is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'y-component velocity' with datatype double,
%%% where w is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'z-component velocity' with datatype double,
%%% note that the icell is full of only ones except for the first layer which is full of twos.
%%% t is just a zero. x, y, and z are linspaces from domainStart+dx/2 
%%% to domainEnd-dx/2 spaced by the corresponding dx

urb_vis_t = ncread(netcdfFile_urb_vis_FlatTerrain,'t');
urb_vis_x = ncread(netcdfFile_urb_vis_FlatTerrain,'x');
urb_vis_y = ncread(netcdfFile_urb_vis_FlatTerrain,'y');
urb_vis_z = ncread(netcdfFile_urb_vis_FlatTerrain,'z');
urb_vis_icell = ncread(netcdfFile_urb_vis_FlatTerrain,'icell');
urb_vis_u = ncread(netcdfFile_urb_vis_FlatTerrain,'u');
urb_vis_v = ncread(netcdfFile_urb_vis_FlatTerrain,'v');
urb_vis_w = ncread(netcdfFile_urb_vis_FlatTerrain,'w');


%% read turb netcdf file

% now get info about the urb file
ncdisp(netcdfFile_turb_FlatTerrain);

%%% ncdisp shows variables t, x, y, z,  iturbflag,  S11, S12, S13, S22, 
%%%  S23, S33, L, tau11, tau12, tau13, tau22, tau23, tau33, CoEps, tke
%%% where t is size 1x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
%%% where x is size 200x1 in dimensions (x) with units 'm' and long_name 'x-distance' with datatype single,
%%% where y is size 200x1 in dimensions (y) with units 'm' and long_name 'y-distance' with datatype single,
%%% where z is size 201x1 in dimensions (z) with units 'm' and long_name 'z-distance' with datatype single,
%%% where iturbflag is size 200x200x201x1 in dimensions (x,y,z,t) with units '--' and long_name 'icell turb flag' with datatype int32,
%%% where S11 is size 200x200x201x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'uu-component of strain-rate tensor' with datatype single,
%%% where S12 is size 200x200x201x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'uv-component of strain-rate tensor' with datatype single,
%%% where S13 is size 200x200x201x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'uw-component of strain-rate tensor' with datatype single,
%%% where S22 is size 200x200x201x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'vv-component of strain-rate tensor' with datatype single,
%%% where S23 is size 200x200x201x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'vw-component of strain-rate tensor' with datatype single,
%%% where S33 is size 200x200x201x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'ww-component of strain-rate tensor' with datatype single,
%%% where L is size 200x200x201x1 in dimensions (x,y,z,t) with units 'm' and long_name 'mixing length' with datatype single,
%%% where tau11 is size 200x200x201x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'uu-component of stress tensor' with datatype single,
%%% where tau12 is size 200x200x201x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'uv-component of stress tensor' with datatype single,
%%% where tau13 is size 200x200x201x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'uw-component of stress tensor' with datatype single,
%%% where tau22 is size 200x200x201x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'vv-component of stress tensor' with datatype single,
%%% where tau23 is size 200x200x201x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'vw-component of stress tensor' with datatype single,
%%% where tau33 is size 200x200x201x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'ww-component of stress tensor' with datatype single,
%%% where CoEps is size 200x200x201x1 in dimensions (x,y,z,t) with units 'm2 s-3' and long_name dissipation rate' with datatype single,
%%% where tke is size 200x200x201x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'turbulent kinetic energy' with datatype single.
%%% t is just a zero. x and y are linspaces from domainStart+dx/2 to domainEnd-dx/2 spaced by the corresponding dx
%%% while z is a linspace from domainStart-dx*3/2 to domainEnd-dx/2 spaced by the corresponding dx

turb_t = ncread(netcdfFile_turb_FlatTerrain,'t');
turb_x = ncread(netcdfFile_turb_FlatTerrain,'x');
turb_y = ncread(netcdfFile_turb_FlatTerrain,'y');
turb_z = ncread(netcdfFile_turb_FlatTerrain,'z');
turb_iturbflag = ncread(netcdfFile_turb_FlatTerrain,'iturbflag');
turb_S11 = ncread(netcdfFile_turb_FlatTerrain,'S11');
turb_S12 = ncread(netcdfFile_turb_FlatTerrain,'S12');
turb_S13 = ncread(netcdfFile_turb_FlatTerrain,'S13');
turb_S22 = ncread(netcdfFile_turb_FlatTerrain,'S22');
turb_S23 = ncread(netcdfFile_turb_FlatTerrain,'S23');
turb_S33 = ncread(netcdfFile_turb_FlatTerrain,'S33');
turb_L = ncread(netcdfFile_turb_FlatTerrain,'L');
turb_tau11 = ncread(netcdfFile_turb_FlatTerrain,'tau11');
turb_tau12 = ncread(netcdfFile_turb_FlatTerrain,'tau12');
turb_tau13 = ncread(netcdfFile_turb_FlatTerrain,'tau13');
turb_tau22 = ncread(netcdfFile_turb_FlatTerrain,'tau22');
turb_tau23 = ncread(netcdfFile_turb_FlatTerrain,'tau23');
turb_tau33 = ncread(netcdfFile_turb_FlatTerrain,'tau33');
turb_CoEps = ncread(netcdfFile_turb_FlatTerrain,'CoEps');
turb_tke = ncread(netcdfFile_turb_FlatTerrain,'tke');


%% read conc netcdf file

% now get info about the urb file
ncdisp(netcdfFile_conc_FlatTerrain);

%%% ncdisp shows variables t, x, y, z,  conc
%%% where t is size 1x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
%%% where x is size 200x1 in dimensions (x) with units 'm' and long_name 'x-distance' with datatype single,
%%% where y is size 200x1 in dimensions (y) with units 'm' and long_name 'y-distance' with datatype single,
%%% where z is size 200x1 in dimensions (z) with units 'm' and long_name 'z-distance' with datatype single,
%%% where conc is size 200x200x200x1 in dimensions (x,y,z,t) with units '#ofPar m-3' and long_name 'concentration' with datatype single.
%%% t is 999 (0 + tavg). x,y, and z are just linspace from domainStart to domainEnd
%%% conc goes from 0 to 3.8574
%%% use max(max(max(data))) or min(min(min(data))) to see such info

conc_t = ncread(netcdfFile_conc_FlatTerrain,'t');
conc_x = ncread(netcdfFile_conc_FlatTerrain,'x');
conc_y = ncread(netcdfFile_conc_FlatTerrain,'y');
conc_z = ncread(netcdfFile_conc_FlatTerrain,'z');
conc_conc = ncread(netcdfFile_conc_FlatTerrain,'conc');


%% read eulerianData netcdf file

% now get info about the urb file
ncdisp(netcdfFile_eulerian_FlatTerrain);

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

eulerian_t = ncread(netcdfFile_eulerian_FlatTerrain,'t');
eulerian_x = ncread(netcdfFile_eulerian_FlatTerrain,'x');
eulerian_y = ncread(netcdfFile_eulerian_FlatTerrain,'y');
eulerian_z = ncread(netcdfFile_eulerian_FlatTerrain,'z');
eulerian_u = ncread(netcdfFile_eulerian_FlatTerrain,'u');
eulerian_v = ncread(netcdfFile_eulerian_FlatTerrain,'v');
eulerian_w = ncread(netcdfFile_eulerian_FlatTerrain,'w');
eulerian_sig_x = ncread(netcdfFile_eulerian_FlatTerrain,'sig_x');
eulerian_sig_y = ncread(netcdfFile_eulerian_FlatTerrain,'sig_y');
eulerian_sig_z = ncread(netcdfFile_eulerian_FlatTerrain,'sig_z');
eulerian_txx = ncread(netcdfFile_eulerian_FlatTerrain,'txx');
eulerian_txy = ncread(netcdfFile_eulerian_FlatTerrain,'txy');
eulerian_txz = ncread(netcdfFile_eulerian_FlatTerrain,'txz');
eulerian_tyy = ncread(netcdfFile_eulerian_FlatTerrain,'tyy');
eulerian_tyz = ncread(netcdfFile_eulerian_FlatTerrain,'tyz');
eulerian_tzz = ncread(netcdfFile_eulerian_FlatTerrain,'tzz');
eulerian_epps = ncread(netcdfFile_eulerian_FlatTerrain,'epps');
eulerian_tke = ncread(netcdfFile_eulerian_FlatTerrain,'tke');
eulerian_dtxxdx = ncread(netcdfFile_eulerian_FlatTerrain,'dtxxdx');
eulerian_dtxydy = ncread(netcdfFile_eulerian_FlatTerrain,'dtxydy');
eulerian_dtxzdz = ncread(netcdfFile_eulerian_FlatTerrain,'dtxzdz');
eulerian_dtxydx = ncread(netcdfFile_eulerian_FlatTerrain,'dtxydx');
eulerian_dtyydy = ncread(netcdfFile_eulerian_FlatTerrain,'dtyydy');
eulerian_dtyzdz = ncread(netcdfFile_eulerian_FlatTerrain,'dtyzdz');
eulerian_dtxzdx = ncread(netcdfFile_eulerian_FlatTerrain,'dtxzdx');
eulerian_dtyzdy = ncread(netcdfFile_eulerian_FlatTerrain,'dtyzdy');
eulerian_dtzzdz = ncread(netcdfFile_eulerian_FlatTerrain,'dtzzdz');
eulerian_flux_div_x = ncread(netcdfFile_eulerian_FlatTerrain,'flux_div_x');
eulerian_flux_div_y = ncread(netcdfFile_eulerian_FlatTerrain,'flux_div_y');
eulerian_flux_div_z = ncread(netcdfFile_eulerian_FlatTerrain,'flux_div_z');


%% read particleInfo netcdf file

% now get info about the urb file
ncdisp(netcdfFile_particle_FlatTerrain);

%%% ncdisp shows variables t, parID,  xPos_init, yPos_init, zPos_init, tStrt, sourceIdx, 
%%%  xPos, yPos, zPos, uFluct, vFluct, wFluct, delta_uFluct, delta_vFluct,
%%%  delta_wFluct, isRogue, isActive
%%% where t is size 100x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
%%% where parID is size 100000x1 in dimensions (parID) with units '--' and long_name 'particle ID' with datatype int32,
%%% where xPos_init is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'initial-x-position' with datatype single,
%%% where yPos_init is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'initial-y-position' with datatype single,
%%% where zPos_init is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'initial-z-position' with datatype single,
%%% where tStrt is size 100000x100 in dimensions (parID,t) with units 's' and long_name 'particle-release-time' with datatype single,
%%% where sourceIdx is size 100000x100 in dimensions (parID,t) with units '--' and long_name 'particle-sourceID' with datatype int32,
%%% where xPos is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'x-position' with datatype single,
%%% where yPos is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'y-position' with datatype single,
%%% where zPos is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'z-position' with datatype single,
%%% where uFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'u-velocity-fluctuation' with datatype single,
%%% where vFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'v-velocity-fluctuation' with datatype single,
%%% where wFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'w-velocity-fluctuation' with datatype single,
%%% where delta_uFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'uFluct-difference' with datatype single,
%%% where delta_vFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'vFluct-difference' with datatype single,
%%% where delta_wFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'wFluct-difference' with datatype single,
%%% where isRogue is size 100000x100 in dimensions (parID,t) with units 'bool' and long_name 'is-particle-rogue' with datatype int32,
%%% where isActive is size 100000x100 in dimensions (parID,t) with units 'bool' and long_name 'is-particle-rogue' with datatype int32.
%%% t is a list from 0 to 990 in increments of 10.

particle_t = ncread(netcdfFile_particle_FlatTerrain,'t');
particle_parID = ncread(netcdfFile_particle_FlatTerrain,'parID');
particle_xPos_init = ncread(netcdfFile_particle_FlatTerrain,'xPos_init');
particle_yPos_init = ncread(netcdfFile_particle_FlatTerrain,'yPos_init');
particle_zPos_init = ncread(netcdfFile_particle_FlatTerrain,'zPos_init');
particle_tStrt = ncread(netcdfFile_particle_FlatTerrain,'tStrt');
particle_sourceIdx = ncread(netcdfFile_particle_FlatTerrain,'sourceIdx');
particle_xPos = ncread(netcdfFile_particle_FlatTerrain,'xPos');
particle_yPos = ncread(netcdfFile_particle_FlatTerrain,'yPos');
particle_zPos = ncread(netcdfFile_particle_FlatTerrain,'zPos');
particle_uFluct = ncread(netcdfFile_particle_FlatTerrain,'uFluct');
particle_vFluct = ncread(netcdfFile_particle_FlatTerrain,'vFluct');
particle_wFluct = ncread(netcdfFile_particle_FlatTerrain,'wFluct');
particle_delta_uFluct = ncread(netcdfFile_particle_FlatTerrain,'delta_uFluct');
particle_delta_vFluct = ncread(netcdfFile_particle_FlatTerrain,'delta_vFluct');
particle_delta_wFluct = ncread(netcdfFile_particle_FlatTerrain,'delta_wFluct');
particle_isRogue = ncread(netcdfFile_particle_FlatTerrain,'isRogue');
particle_isActive = ncread(netcdfFile_particle_FlatTerrain,'isActive');




