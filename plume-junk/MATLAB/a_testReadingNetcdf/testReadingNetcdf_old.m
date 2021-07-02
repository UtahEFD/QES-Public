

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


netcdfFile_urb_FlatTerrain = sprintf('%s/FlatTerrain_urb_old.nc',workingDir);
netcdfFile_turb_FlatTerrain = sprintf('%s/FlatTerrain_turb_old.nc',workingDir);
netcdfFile_plume_FlatTerrain = sprintf('%s/FlatTerrain_plume_old.nc',workingDir);


%% read urb netcdf file

% now get info about the urb file
ncdisp(netcdfFile_urb_FlatTerrain);

%%% ncdisp shows variables t, x, y, z, terrain, u, v, w, icell
%%% where t is size 1x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
%%% where x is size 200x1 in dimensions (x) with units 'm' and long_name 'x-distance' with datatype double,
%%% where y is size 200x1 in dimensions (y) with units 'm' and long_name 'y-distance' with datatype double,
%%% where z is size 200x1 in dimensions (z) with units 'm' and long_name 'z-distance' with datatype double,
%%% where terrain is size 200x200 in dimensions (x,y) with units 'm' and long_name 'terrain height' with datatype double,
%%% where u is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'x-component velocity' with datatype double,
%%% where v is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'y-component velocity' with datatype double,
%%% where w is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'z-component velocity' with datatype double,
%%% where icell is size 200x200x200x1 in dimensions (x,y,z,t) with units '--' and long_name 'icell flag value' with datatype int32.
%%% note that the terrain is full of straight zeros and the icell is full
%%% of only ones. t is just a zero. Looks like u has values from 1.9412 to
%%% 8.2326 m/s. v and w are straight zeros. x, y, and z just have the
%%% linspace from domainStart to domainEnd
urb_t = ncread(netcdfFile_urb_FlatTerrain,'t');
urb_x = ncread(netcdfFile_urb_FlatTerrain,'x');
urb_y = ncread(netcdfFile_urb_FlatTerrain,'y');
urb_z = ncread(netcdfFile_urb_FlatTerrain,'z');
urb_terrain = ncread(netcdfFile_urb_FlatTerrain,'terrain');
urb_u = ncread(netcdfFile_urb_FlatTerrain,'u');
urb_u_units = ncreadatt(netcdfFile_urb_FlatTerrain,'u','units');
urb_u_long_name = ncreadatt(netcdfFile_urb_FlatTerrain,'u','long_name');
urb_v = ncread(netcdfFile_urb_FlatTerrain,'v');
urb_w = ncread(netcdfFile_urb_FlatTerrain,'w');
urb_icell = ncread(netcdfFile_urb_FlatTerrain,'icell');


%% read turb netcdf file

% now get info about the urb file
ncdisp(netcdfFile_turb_FlatTerrain);

%%% ncdisp shows variables t, x, y, z, terrain, u, v, w, icell
%%% where t is size 1x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
%%% where x is size 200x1 in dimensions (x) with units 'm' and long_name 'x-distance' with datatype double,
%%% where y is size 200x1 in dimensions (y) with units 'm' and long_name 'y-distance' with datatype double,
%%% where z is size 200x1 in dimensions (z) with units 'm' and long_name 'z-distance' with datatype double,
%%% where tau_11 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'kg m-1 s-2' and long_name 'uu stress' with datatype double,
%%% where tau_12 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'kg m-1 s-2' and long_name 'uv stress' with datatype double,
%%% where tau_13 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'kg m-1 s-2' and long_name 'uw stress' with datatype double,
%%% where tau_22 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'kg m-1 s-2' and long_name 'vv stress' with datatype double,
%%% where tau_23 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'kg m-1 s-2' and long_name 'vw stress' with datatype double,
%%% where tau_33 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'kg m-1 s-2' and long_name 'ww stress' with datatype double,
%%% where sig_11 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'uu sigma' with datatype double,
%%% where sig_22 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'vv sigma' with datatype double,
%%% where sig_33 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s-1' and long_name 'ww sigma' with datatype double,
%%% where lam_11 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s2 kg-1' and long_name 'uu lambda' with datatype double,
%%% where lam_12 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s2 kg-1' and long_name 'uv lambda' with datatype double,
%%% where lam_13 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s2 kg-1' and long_name 'uw lambda' with datatype double,
%%% where lam_21 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s2 kg-1' and long_name 'vu lambda' with datatype double,
%%% where lam_22 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s2 kg-1' and long_name 'vv lambda' with datatype double,
%%% where lam_23 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s2 kg-1' and long_name 'vw lambda' with datatype double,
%%% where lam_31 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s2 kg-1' and long_name 'wu lambda' with datatype double,
%%% where lam_32 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s2 kg-1' and long_name 'wv lambda' with datatype double,
%%% where lam_33 is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm s2 kg-1' and long_name 'ww lambda' with datatype double,
%%% where dudx is size 200x200x200x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'dudx' with datatype double,
%%% where dudy is size 200x200x200x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'dudy' with datatype double,
%%% where dudz is size 200x200x200x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'dudz' with datatype double,
%%% where dvdx is size 200x200x200x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'dvdx' with datatype double,
%%% where dvdy is size 200x200x200x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'dvdy' with datatype double,
%%% where dvdz is size 200x200x200x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'dvdz' with datatype double,
%%% where dwdx is size 200x200x200x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'dwdx' with datatype double,
%%% where dwdy is size 200x200x200x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'dwdy' with datatype double,
%%% where dwdz is size 200x200x200x1 in dimensions (x,y,z,t) with units 's-1' and long_name 'dwdz' with datatype double,
%%% where CoEps is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-3' and long_name 'CoEps' with datatype double,
%%% where tke is size 200x200x200x1 in dimensions (x,y,z,t) with units 'm2 s-2' and long_name 'tke' with datatype double.
%%% t is zero and x,y, and z go from domainStart to domainEnd
%%% tau_11 goes from 0 to 0.0573, tau_12, tau_13, and tau_23 are straight zeros,
%%% tau_22 goes from 0 to 0.0707, tau_33 goes from 0 to 0.0299
%%% sig_11 goes from 0 to 0.2393, sig_22 goes from 0 to 0.2659, 
%%% sig_33 goes from 0 to 0.1728
%%% lam_11 goes from 0 to 6.2845e5, lam_13 goes from 0 to 5.5779e5, lam_22
%%% goes from 0 to 2.9987e5, lam_31 goes from 0 to 5.5779e5, lam_33 goes
%%% from 0 to 1.2048e6
%%% lam_12, lam_21, lam_23, and lam_32 are straight zeros
%%% dudx, dvdx, dwdx, dudy, dvdy, dwdy, dvdz, and dwdz  are straight zeros
%%% dudz goes from 0 to 0.7943
%%% CoEps goes from 0 to 0.12 tke goes from 0 to 0.0877
turb_t = ncread(netcdfFile_turb_FlatTerrain,'t');
turb_x = ncread(netcdfFile_turb_FlatTerrain,'x');
turb_y = ncread(netcdfFile_turb_FlatTerrain,'y');
turb_z = ncread(netcdfFile_turb_FlatTerrain,'z');
turb_tau_11 = ncread(netcdfFile_turb_FlatTerrain,'tau_11');
turb_tau_12 = ncread(netcdfFile_turb_FlatTerrain,'tau_12');
turb_tau_13 = ncread(netcdfFile_turb_FlatTerrain,'tau_13');
turb_tau_22 = ncread(netcdfFile_turb_FlatTerrain,'tau_22');
turb_tau_23 = ncread(netcdfFile_turb_FlatTerrain,'tau_23');
turb_tau_33 = ncread(netcdfFile_turb_FlatTerrain,'tau_33');
turb_sig_11 = ncread(netcdfFile_turb_FlatTerrain,'sig_11');
turb_sig_22 = ncread(netcdfFile_turb_FlatTerrain,'sig_22');
turb_sig_33 = ncread(netcdfFile_turb_FlatTerrain,'sig_33');
turb_lam_11 = ncread(netcdfFile_turb_FlatTerrain,'lam_11');
turb_lam_12 = ncread(netcdfFile_turb_FlatTerrain,'lam_12');
turb_lam_13 = ncread(netcdfFile_turb_FlatTerrain,'lam_13');
turb_lam_21 = ncread(netcdfFile_turb_FlatTerrain,'lam_21');
turb_lam_22 = ncread(netcdfFile_turb_FlatTerrain,'lam_22');
turb_lam_23 = ncread(netcdfFile_turb_FlatTerrain,'lam_23');
turb_lam_31 = ncread(netcdfFile_turb_FlatTerrain,'lam_31');
turb_lam_32 = ncread(netcdfFile_turb_FlatTerrain,'lam_32');
turb_lam_33 = ncread(netcdfFile_turb_FlatTerrain,'lam_33');
turb_dudx = ncread(netcdfFile_turb_FlatTerrain,'dudx');
turb_dudy = ncread(netcdfFile_turb_FlatTerrain,'dudy');
turb_dudz = ncread(netcdfFile_turb_FlatTerrain,'dudz');
turb_dvdx = ncread(netcdfFile_turb_FlatTerrain,'dvdx');
turb_dvdy = ncread(netcdfFile_turb_FlatTerrain,'dvdy');
turb_dvdz = ncread(netcdfFile_turb_FlatTerrain,'dvdz');
turb_dwdx = ncread(netcdfFile_turb_FlatTerrain,'dwdx');
turb_dwdy = ncread(netcdfFile_turb_FlatTerrain,'dwdy');
turb_dwdz = ncread(netcdfFile_turb_FlatTerrain,'dwdz');
turb_CoEps = ncread(netcdfFile_turb_FlatTerrain,'CoEps');
turb_tke = ncread(netcdfFile_turb_FlatTerrain,'tke');


%% read plume netcdf file

% now get info about the urb file
ncdisp(netcdfFile_plume_FlatTerrain);

%%% ncdisp shows variables t, x, y, z, terrain, u, v, w, icell
%%% where t is size 1x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
%%% where x is size 200x1 in dimensions (x) with units 'm' and long_name 'x-distance' with datatype double,
%%% where y is size 200x1 in dimensions (y) with units 'm' and long_name 'y-distance' with datatype double,
%%% where z is size 200x1 in dimensions (z) with units 'm' and long_name 'z-distance' with datatype double,
%%% where conc is size 200x200x200x1 in dimensions (x,y,z,t) with units '--' and long_name 'concentration' with datatype double.
%%% t is zero. x,y, and z are just linspace from domainStart to domainEnd
%%% conc goes from 0 to 2.3023e-8
%%% use max(max(max(data))) or min(min(min(data))) to see such info
plume_t = ncread(netcdfFile_plume_FlatTerrain,'t');
plume_x = ncread(netcdfFile_plume_FlatTerrain,'x');
plume_y = ncread(netcdfFile_plume_FlatTerrain,'y');
plume_z = ncread(netcdfFile_plume_FlatTerrain,'z');
plume_conc = ncread(netcdfFile_plume_FlatTerrain,'conc');







