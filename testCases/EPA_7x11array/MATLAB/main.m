% Uniform Flow test case for QES-plume
% Base on Singh PhD Dissertation )
% Initial test case published in 
%  Singh et al. 2004 
%  Willemsen et al. 2007
%
% F. Margaiaraz
% Univesity of Utah. 2021
clear all

H=15; % m
U=2.83; % m/s 

% source info
Q=200; % #par/s (source strength)
tRelease=2000; % total time of release
Ntot=Q*tRelease; % total number of particles

% concentration info
dt=1; % s
tAvg=1800; % s 

fsize=12;

xS=50;yS=200;zS=1;

% set the case base name for use in all the other file paths
casePath="../QES-data";

caseNameWinds = "7x11array_shapefile";
caseNamePlume = "7x11array_shapefile";

% set the plotOutputFolders
plotOutputDir = "plotOutput2";
mkdir(plotOutputDir)

data=struct();
varnames=struct();

% read wind netcdf file
fileName = sprintf("%s/%s_windsOut.nc",casePath,caseNameWinds);
[data.winds,varnames.winds] = readNetCDF(fileName);
% read turb netcdf file
fileName = sprintf("%s/%s_turbOut.nc",casePath,caseNameWinds);
[data.turb,varnames.turb] = readNetCDF(fileName);

% read main plume files
fileName = sprintf("%s/%s_plumeOut.nc",casePath,caseNamePlume);
[data.plume,varnames.plume] = readNetCDF(fileName);

xoH=(data.plume.x-xS)/H;
yoH=(data.plume.y-yS)/H;
zoH=(data.plume.z)/H;

boxDx=mean(diff(data.plume.x));
boxDy=mean(diff(data.plume.y));
boxDz=mean(diff(data.plume.z));
boxVol=double(boxDx*boxDy*boxDz);
CC=dt/tAvg/boxVol;

idy=find(yoH >= 0,1);
plot_ySlice_pcolor

idy=find(yoH >= .2,1);
plot_ySlice_pcolor

idy=find(yoH > 1,1);
plot_ySlice_pcolor

idz=find(zoH >= 0.25,1);
plot_zSlice_pcolor

idz=find(zoH >= 0.50,1);
plot_zSlice_pcolor

idz=find(zoH >= 0.75,1);
plot_zSlice_pcolor

idz=find(zoH > 1.00,1);
plot_zSlice_pcolor

idz=find(zoH >= 1.25,1);
plot_zSlice_pcolor
