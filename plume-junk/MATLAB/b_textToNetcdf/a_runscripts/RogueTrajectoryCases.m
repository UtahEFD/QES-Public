

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


%%% make sure to grab all functions from all folders
workingDir = pwd;
%%%% TODO: for now put it all in one folder, but at some point in time
%%%% separate out into different folders
mainCodeDir = sprintf("%s/../b_src",workingDir);

functionsDir = sprintf("%s/functions",mainCodeDir);
addpath(mainCodeDir,  functionsDir);


%%%%%%% end standard main function stuff


%% sinewave case

inputDir = sprintf('%s/../c_textInputs/a_sinewave',workingDir);
outputDir = sprintf('%s/../d_netcdfOutputs/a_sinewave',workingDir);
outputBaseName = 'BaileySinewave';


x_turb_domainStart = 1;
x_turb_domainEnd = 1;
y_turb_domainStart = 1;
y_turb_domainEnd = 1;
z_turb_domainStart = 0;
z_turb_domainEnd = 1;

x_turb_nCells = 1;
y_turb_nCells = 1;
z_turb_nCells = 20;

x_turb_BCtype = "";
y_turb_BCtype = "";
z_turb_BCtype = "periodic";

C0 = 4.0;
epps_file = sprintf('%s/sinewave_epps.txt',inputDir);
sigma2_file = sprintf('%s/sinewave_sigma2.txt',inputDir);
txx_file = sprintf('%s/sinewave_txx.txt',inputDir);
txy_file = sprintf('%s/sinewave_txy.txt',inputDir);
txz_file = sprintf('%s/sinewave_txz.txt',inputDir);
tyy_file = sprintf('%s/sinewave_tyy.txt',inputDir);
tyz_file = sprintf('%s/sinewave_tyz.txt',inputDir);
tzz_file = sprintf('%s/sinewave_tzz.txt',inputDir);


x_urb_domainStart = 1;
x_urb_domainEnd = 1;
y_urb_domainStart = 1;
y_urb_domainEnd = 1;
z_urb_domainStart = 0;
z_urb_domainEnd = 1;

x_urb_nCells = 1;
y_urb_nCells = 1;
z_urb_nCells = 20;

x_urb_BCtype = "";
y_urb_BCtype = "";
z_urb_BCtype = "periodic";

uMean_file = "";
vMean_file = "";
wMean_file = "";


% now finally run the function
textToNetcdf(outputDir,outputBaseName,  x_turb_domainStart,x_turb_domainEnd,y_turb_domainStart,y_turb_domainEnd,z_turb_domainStart,z_turb_domainEnd, x_turb_nCells,y_turb_nCells,z_turb_nCells, x_turb_BCtype,y_turb_BCtype,z_turb_BCtype, C0,epps_file,sigma2_file,txx_file,txy_file,txz_file,tyy_file,tyz_file,tzz_file,  x_urb_domainStart,x_urb_domainEnd,y_urb_domainStart,y_urb_domainEnd,z_urb_domainStart,z_urb_domainEnd, x_urb_nCells,y_urb_nCells,z_urb_nCells, x_urb_BCtype,y_urb_BCtype,z_urb_BCtype, uMean_file,vMean_file,wMean_file);


%% channel case

inputDir = sprintf('%s/../c_textInputs/b_channel',workingDir);
outputDir = sprintf('%s/../d_netcdfOutputs/b_channel',workingDir);
outputBaseName = 'BaileyChannel';


x_turb_domainStart = 1;
x_turb_domainEnd = 1;
y_turb_domainStart = 1;
y_turb_domainEnd = 1;
z_turb_domainStart = 0;
z_turb_domainEnd = 1;

x_turb_nCells = 1;
y_turb_nCells = 1;
z_turb_nCells = 50;

x_turb_BCtype = "";
y_turb_BCtype = "";
z_turb_BCtype = "reflection";

C0 = 4.0;
epps_file = sprintf('%s/channel_epps.txt',inputDir);
sigma2_file = sprintf('%s/channel_sigma2.txt',inputDir);
txx_file = sprintf('%s/channel_txx.txt',inputDir);
txy_file = sprintf('%s/channel_txy.txt',inputDir);
txz_file = sprintf('%s/channel_txz.txt',inputDir);
tyy_file = sprintf('%s/channel_tyy.txt',inputDir);
tyz_file = sprintf('%s/channel_tyz.txt',inputDir);
tzz_file = sprintf('%s/channel_tzz.txt',inputDir);


x_urb_domainStart = 1;
x_urb_domainEnd = 1;
y_urb_domainStart = 1;
y_urb_domainEnd = 1;
z_urb_domainStart = 0;
z_urb_domainEnd = 1;

x_urb_nCells = 1;
y_urb_nCells = 1;
z_urb_nCells = 50;

x_urb_BCtype = "";
y_urb_BCtype = "";
z_urb_BCtype = "reflection";

uMean_file = sprintf('%s/channel_u.txt',inputDir);
vMean_file = "";
wMean_file = "";


% now finally run the function
textToNetcdf(outputDir,outputBaseName,  x_turb_domainStart,x_turb_domainEnd,y_turb_domainStart,y_turb_domainEnd,z_turb_domainStart,z_turb_domainEnd, x_turb_nCells,y_turb_nCells,z_turb_nCells, x_turb_BCtype,y_turb_BCtype,z_turb_BCtype, C0,epps_file,sigma2_file,txx_file,txy_file,txz_file,tyy_file,tyz_file,tzz_file,  x_urb_domainStart,x_urb_domainEnd,y_urb_domainStart,y_urb_domainEnd,z_urb_domainStart,z_urb_domainEnd, x_urb_nCells,y_urb_nCells,z_urb_nCells, x_urb_BCtype,y_urb_BCtype,z_urb_BCtype, uMean_file,vMean_file,wMean_file);


%% LES case

inputDir = sprintf('%s/../c_textInputs/c_LES',workingDir);
outputDir = sprintf('%s/../d_netcdfOutputs/c_LES',workingDir);
outputBaseName = 'BaileyLES';


x_turb_domainStart = 0;
x_turb_domainEnd = 6393.2;
y_turb_domainStart = 0;
y_turb_domainEnd = 6283.2;
z_turb_domainStart = 0;
z_turb_domainEnd = 1000;

x_turb_nCells = 32;
y_turb_nCells = 32;
z_turb_nCells = 32;

x_turb_BCtype = "periodic";
y_turb_BCtype = "periodic";
z_turb_BCtype = "reflection";

C0 = 4.0;
epps_file = sprintf('%s/LES_epps.txt',inputDir);
sigma2_file = sprintf('%s/LES_sigma2.txt',inputDir);
txx_file = sprintf('%s/LES_txx.txt',inputDir);
txy_file = sprintf('%s/LES_txy.txt',inputDir);
txz_file = sprintf('%s/LES_txz.txt',inputDir);
tyy_file = sprintf('%s/LES_tyy.txt',inputDir);
tyz_file = sprintf('%s/LES_tyz.txt',inputDir);
tzz_file = sprintf('%s/LES_tzz.txt',inputDir);


x_urb_domainStart = 0;
x_urb_domainEnd = 6393.2;
y_urb_domainStart = 0;
y_urb_domainEnd = 6283.2;
z_urb_domainStart = 0;
z_urb_domainEnd = 1000;

x_urb_nCells = 32;
y_urb_nCells = 32;
z_urb_nCells = 32;

x_urb_BCtype = "periodic";
y_urb_BCtype = "periodic";
z_urb_BCtype = "reflection";

uMean_file = sprintf('%s/LES_u.txt',inputDir);
vMean_file = sprintf('%s/LES_v.txt',inputDir);
wMean_file = sprintf('%s/LES_w.txt',inputDir);


% now finally run the function
textToNetcdf(outputDir,outputBaseName,  x_turb_domainStart,x_turb_domainEnd,y_turb_domainStart,y_turb_domainEnd,z_turb_domainStart,z_turb_domainEnd, x_turb_nCells,y_turb_nCells,z_turb_nCells, x_turb_BCtype,y_turb_BCtype,z_turb_BCtype, C0,epps_file,sigma2_file,txx_file,txy_file,txz_file,tyy_file,tyz_file,tzz_file,  x_urb_domainStart,x_urb_domainEnd,y_urb_domainStart,y_urb_domainEnd,z_urb_domainStart,z_urb_domainEnd, x_urb_nCells,y_urb_nCells,z_urb_nCells, x_urb_BCtype,y_urb_BCtype,z_urb_BCtype, uMean_file,vMean_file,wMean_file);





