

%%%%%%% standard main function stuff


% clear old stuff
% clc;
% clear;
% close all;
% path(pathdef);  % resets the path

% return to current directory
% mfile_name          = mfilename('fullpath');
% [pathstr,name,ext]  = fileparts(mfile_name);
% cd(pathstr);


%%% make sure to grab all functions from all folders
% workingDir = pwd;
% 
% srcDir = sprintf("%s/src",workingDir);

% separate functions inside the src dir
% netCDFfunctionsDir = sprintf("%s/netCDFfunctions",srcDir);
% plotFunctionsDir = sprintf("%s/plotFunctions",srcDir);
% 
% addpath(srcDir,  netCDFfunctionsDir,plotFunctionsDir);


%%%%%%% end standard main function stuff

% now start doing the code stuff


%%% set the case base name for use in all the other file paths
caseBaseName = "Channel3D";

Ttot=18.3;
dt=[1.83,0.183,0.00183];

caseNamesFile=strings(numel(dt),1);
caseNames=strings(numel(dt),1);
for k=1:numel(dt)
    caseNamesFile(k)=sprintf('%s_%G_%G',caseBaseName,dt(k),Ttot);
    dtStr=strrep(sprintf('%G',dt(k)),'.','o');
    TStr=strrep(sprintf('%G',Ttot),'.','o');
    caseNames(k)=sprintf('%s_%s_%s',caseBaseName,dtStr,TStr);
end
nCases=numel(caseNamesFile);

%%% set the input urb netcdf file
windsInputFile = sprintf("../QES-data/%s_windsWk.nc",caseBaseName);

%%% set the input turb netcdf file
turbInputFile = sprintf("../QES-data/%s_turbOut.nc",caseBaseName);


%%% set the plume output file paths
% plumeOutputFolder = sprintf("%s/../..",workingDir);
% caseFolders = [
%     
%     "a_0o01_10";      % 1
%     "b_0o05_10";      % 2
%     "c_0o1_10";       % 3
%     "d_4_10";         % 4
%     
%     ];
% nCaseFolders = length(caseFolders);


%%% set the input concentration files
% plumeConcInputFiles = strings(nCase,1);
% for k = 1:nCaseFolders
%     plumeConcInputFiles(k) = sprintf("%s/%s/%s_conc.nc",plumeOutputFolder,caseFolders(k),caseBaseName);
% end


%%% set the input particleInfo files
% plumeParInfoInputFiles = strings(nCaseFolders,1);
% for k = 1:nCaseFolders
%     plumeParInfoInputFiles(k) = sprintf("%s/%s/%s_particleInfo.nc",plumeOutputFolder,caseFolders(k),caseBaseName);
% end




%%% set the plotOutputFolders
plotOutputDir = "plotOutput";
mkdir(plotOutputDir)

%%% set the universal constant
C0 = 4.0;


%%% set additional plot variables necessary for the plots
nProbabilityBins = 25;
probabilityLim = [0 2.5];

nStatisticBins = 20;
wFluct_averages_Lim = [-0.04, 0.04];
wFluct_variances_Lim = [0, 0.04];
delta_wFluct_averages_Lim = [-0.2 0.2];
delta_wFluct_variances_Lim = [0 0.08];



% now take the caseFolderNames and turn them into caseNames
[ ~, timestep_array, currentTime_array ] = setCaseInfo(caseNames,caseBaseName);


% now input the urb and turb files
[winds_data,winds_varnames] = readNetCDF(windsInputFile);
[turb_data,turb_varnames] = readNetCDF(turbInputFile);


% use the input cell centered grid values to calculate all the grid
% information
[ xGridInfo, yGridInfo, zGridInfo ] = calcGridInfo( winds_data.x_cc, winds_data.y_cc, winds_data.z_cc );


% now input the conc and particleInfo files
for k = 1:nCases
    fileName = sprintf("../QES-data/%s_conc.nc",caseNamesFile(k));
    [plumeConc_data.(caseNames(k)),plumeConc_varnames.(caseNames(k))] = readNetCDF(fileName);
    fileName = sprintf("../QES-data/%s_particleInfo.nc",caseNamesFile(k));
    [plumeParInfo_data.(caseNames(k)),plumeParInfo_varnames.(caseNames(k))] = readNetCDF(fileName);
end



%%% first do all the plots with show_ghostCell as false
show_ghostCells = false;

% now plot the probability plot
plotProbabilityFigure(caseBaseName,plotOutputDir,probabilityLim,nProbabilityBins,...
    currentTime_array,timestep_array,zGridInfo.Lz,caseNames,plumeParInfo_data);


% now the statistics plot
plotStatisticsFigure( caseBaseName,plotOutputDir,show_ghostCells,...
    wFluct_averages_Lim,wFluct_variances_Lim,delta_wFluct_averages_Lim,delta_wFluct_variances_Lim,...
    nStatisticBins,currentTime_array,timestep_array,zGridInfo,...
    turb_data,caseNames,plumeParInfo_data,C0);


%%% now do all the plots with show_ghostCell as true
show_ghostCells = true;

% now the statistics plot
plotStatisticsFigure(caseBaseName,plotOutputDir,show_ghostCells,...
    wFluct_averages_Lim,wFluct_variances_Lim,delta_wFluct_averages_Lim,delta_wFluct_variances_Lim,...
    nStatisticBins,currentTime_array,timestep_array,zGridInfo,...
    turb_data,caseNames,plumeParInfo_data,C0);








