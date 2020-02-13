

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
mainCodeDir = sprintf("%s/../../b_src/e_plotResultFunctions",workingDir);
sharedFunctionsDir = sprintf("%s/../ff_sharedFunctions",mainCodeDir);

%specificFunctionsDir = sprintf("%s/functions",mainCodeDir);
%addpath(mainCodeDir,sharedFunctionsDir,  specificFunctionsDir);
addpath(mainCodeDir,sharedFunctionsDir );


%%%%%%% end standard main function stuff


%% sinewave case


%%% set the input file directory (codeInputDir) as well as the file base names
caseBaseName = "channel";
baseCodeInputDir = sprintf("%s/../../../../../testCases/Bailey/b_channel/c_plumeOutputs",workingDir);
codeInputFolders = [
    
    "a_0o00183_18o3";       % 1
    "b_0o183_18o3";         % 2
    "c_1o83_18o3";          % 3
    
    ];
nCodeInputFiles = length(codeInputFolders);

% setup the codeInputDirs from the base directory and the individual
% directories
codeInputDirs = strings(nCodeInputFiles,1);
for fileIdx = 1:nCodeInputFiles
    codeInputDirs(fileIdx) = sprintf("%s/%s",baseCodeInputDir,codeInputFolders(fileIdx));
end

folderFileNames = [
    
    "sim_info.txt";      % 1
    
    sprintf("%s_eulerianData.nc",caseBaseName);    % 2
    sprintf("%s_conc.nc",caseBaseName);            % 3
    sprintf("%s_particleInfo.nc",caseBaseName);    % 4
    
    ];

%%% now stuff all those input information into the codeInputFiles variable
codeInputFiles = cell(2,1);
codeInputFiles(1) = {codeInputDirs};
codeInputFiles(2) = {folderFileNames};


%%% set the plotOutputDir
plotOutputDir = sprintf("%s/../../../../../testCases/Bailey/b_channel/e_matlabPlotOutput/e_resultPlots",workingDir);



%%% now do some coding, using each of the specific functions to generate
%%% whatever plot you want


% now load in data
[fileExists_array, saveBasename_array,  current_time_array,timestep_array,  C_0_array,nParticles_array,  xCellGrid_array,yCellGrid_array,zCellGrid_array,  uMean_data_array,vMean_data_array,wMean_data_array,sigma2_data_array,epps_data_array,txx_data_array,txy_data_array,txz_data_array,tyy_data_array,tyz_data_array,tzz_data_array,  dtxxdx_data_array,dtxydx_data_array,dtxzdx_data_array,dtxydy_data_array,dtyydy_data_array,dtyzdy_data_array,dtxzdz_data_array,dtyzdz_data_array,dtzzdz_data_array,  flux_div_x_data_array,flux_div_y_data_array,flux_div_z_data_array,  uFluct_array,vFluct_array,wFluct_array,delta_uFluct_array,delta_vFluct_array,delta_wFluct_array,  rogueCount_array,isActive_array,  xPos_array,yPos_array,zPos_array] = loadCodeOutput(codeInputFiles);

% take the saveBasename_array and create the plotBasename_array
[plotBasename_array] = saveToPlotBasename(fileExists_array,saveBasename_array);



% now plot fig12 stuff
% this is heterogeneous anisotropic positions
% is repeating the plots for the LES case for this case
%%% set the index of overall list of values paths for each plot of interest
fig12plotOutputIndices = [3,2,1];
%%% set figure plot filenames
fig12filename = sprintf("%s/channel_briansFig12Plot.png",plotOutputDir);
%%% set other needed values
nSubPlotRows = 1;
nSubPlotCols = 3;
subPlotIndices = 1:3;
figTitle = "Brians Paper Fig 12";
dimName = "z";
nParticleBins = 25;
probabilityLim = [0 2.5];
plotProbabilityFigure(nSubPlotRows,nSubPlotCols,subPlotIndices,  figTitle,dimName,probabilityLim,nParticleBins,  fileExists_array(fig12plotOutputIndices), plotBasename_array(fig12plotOutputIndices),   current_time_array(fig12plotOutputIndices),timestep_array(fig12plotOutputIndices),nParticles_array(fig12plotOutputIndices),  rogueCount_array(fig12plotOutputIndices),isActive_array(fig12plotOutputIndices),  zPos_array(fig12plotOutputIndices));
% get the current figure handle for saving the figure
fig = gcf;
saveas(fig,fig12filename);
saveas(fig,strrep(fig12filename,'.png','.fig'));
% delete current figure. Pause before and after to make sure no errors
% occur in other processes
pause(3);
close(fig);
pause(3);


% now plot fig15 stuff, a made up plot to repeat for each case
% this is the heterogeneous anisotropic regular statistics
% is repeating the plots for the LES case for this case
%%% set the index of overall list of values paths for each plot of interest
fig15plotOutputIndices = [1,2,3];
%%% set figure plot filenames
fig15filename = sprintf("%s/sinewave_briansFig15Plot.png",plotOutputDir);
%%% set other needed values
figTitle = "Brians Paper Fig 15";
dimName = "z";
nStatisticBins = 20;
velFluct_averages_Lim = [-0.04, 0.04];
velFluct_variances_Lim = [0, 0.04];
delta_velFluct_averages_Lim = [-0.2, 0.8];
delta_velFluct_variances_Lim = [0, 0.08];
expectedValueIdx = 3;
plotStatisticsFigure(figTitle,dimName,nStatisticBins,  velFluct_averages_Lim,velFluct_variances_Lim,delta_velFluct_averages_Lim,delta_velFluct_variances_Lim,  fileExists_array(fig15plotOutputIndices),  plotBasename_array(fig15plotOutputIndices),  cell2mat(C_0_array(expectedValueIdx)),cell2mat(tzz_data_array(expectedValueIdx)),cell2mat(dtzzdz_data_array(expectedValueIdx)),cell2mat(epps_data_array(expectedValueIdx)),cell2mat(xCellGrid_array(expectedValueIdx)),cell2mat(yCellGrid_array(expectedValueIdx)),cell2mat(zCellGrid_array(expectedValueIdx)),fileExists_array(expectedValueIdx),  current_time_array(fig15plotOutputIndices),timestep_array(fig15plotOutputIndices),  isActive_array(fig15plotOutputIndices),  zPos_array(fig15plotOutputIndices),wFluct_array(fig15plotOutputIndices),delta_wFluct_array(fig15plotOutputIndices));
% get the current figure handle for saving the figure
fig = gcf;
saveas(fig,fig15filename);
saveas(fig,strrep(fig15filename,'.png','.fig'));
% delete current figure. Pause before and after to make sure no errors
% occur in other processes
pause(3);
close(fig);
pause(3);


% now plot fig 13 stuff
% this is the heterogeneous anisotropic other statistics (covariance)
% is repeating the plots for the LES case for this case
%%% set the index of overall list of values paths for each plot of interest
fig13plotOutputIndices = [1,2,3];
%%% set figure plot filenames
fig13filename = sprintf("%s/briansFig13Plot.png",plotOutputDir);
%%% set other needed values
figTitle = "Brians Paper Fig 13";
dimName = "z";
nStatisticBins = 100;
zLim = "";
subfilter_tke_Lim = "";
uFluct_wFluct_covariances_Lim = "";
delta_wFluct_variances_Lim = "";
expectedValueIdx = 3;
plotStatisticsFigure_LES(figTitle,dimName,nStatisticBins,  zLim,subfilter_tke_Lim,uFluct_wFluct_covariances_Lim,delta_wFluct_variances_Lim,  fileExists_array(fig13plotOutputIndices),  plotBasename_array(fig13plotOutputIndices),  cell2mat(C_0_array(expectedValueIdx)),cell2mat(sigma2_data_array(expectedValueIdx)),cell2mat(txz_data_array(expectedValueIdx)),cell2mat(epps_data_array(expectedValueIdx)),cell2mat(xCellGrid_array(expectedValueIdx)),cell2mat(yCellGrid_array(expectedValueIdx)),cell2mat(zCellGrid_array(expectedValueIdx)),fileExists_array(expectedValueIdx),  current_time_array(fig13plotOutputIndices),timestep_array(fig13plotOutputIndices),  isActive_array(fig13plotOutputIndices),  zPos_array(fig13plotOutputIndices),uFluct_array(fig13plotOutputIndices),wFluct_array(fig13plotOutputIndices),delta_wFluct_array(fig13plotOutputIndices));
% get the current figure handle for saving the figure
fig = gcf;
saveas(fig,fig13filename);
saveas(fig,strrep(fig13filename,'.png','.fig'));
% delete current figure. Pause before and after to make sure no errors
% occur in other processes
pause(3);
close(fig);
pause(3);

