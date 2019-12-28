

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
baseCodeInputDir = sprintf("%s/../../c_plumeOutput/b_channel",workingDir);
codeInputFolders = [
    
    "a_HeteroAnisoExplicitTurb_0o00183_18o3";       % 1
    "b_HeteroAnisoExplicitTurb_0o183_18o3";         % 2
    "c_HeteroAnisoExplicitTurb_1o83_18o3";          % 3
    
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
    
    "urb_xCellGrid.txt";    % 2
    "urb_yCellGrid.txt";    % 3
    "urb_zCellGrid.txt";    % 4
    
    "eulerian_uMean.txt";       % 5
    "eulerian_vMean.txt";       % 6
    "eulerian_wMean.txt";       % 7
    "eulerian_sigma2.txt";      % 8
    "eulerian_epps.txt";        % 9
    "eulerian_txx.txt";        % 10
    "eulerian_txy.txt";        % 11
    "eulerian_txz.txt";        % 12
    "eulerian_tyy.txt";        % 13
    "eulerian_tyz.txt";        % 14
    "eulerian_tzz.txt";        % 15
    
    "eulerian_dtxxdx.txt";      % 16
    "eulerian_dtxydx.txt";      % 17
    "eulerian_dtxzdx.txt";      % 18
    "eulerian_dtxydy.txt";      % 19
    "eulerian_dtyydy.txt";      % 20
    "eulerian_dtyzdy.txt";      % 21
    "eulerian_dtxzdz.txt";      % 22
    "eulerian_dtyzdz.txt";      % 23
    "eulerian_dtzzdz.txt";      % 24
    
    "eulerian_flux_div_x.txt";      % 25
    "eulerian_flux_div_y.txt";      % 26
    "eulerian_flux_div_z.txt";      % 27
    
    
    "particle_txx_old.txt";      % 28
    "particle_txy_old.txt";      % 29
    "particle_txz_old.txt";      % 30
    "particle_tyy_old.txt";      % 31
    "particle_tyz_old.txt";      % 32
    "particle_tzz_old.txt";      % 33
    "particle_uFluct_old.txt";      % 34
    "particle_vFluct_old.txt";      % 35
    "particle_wFluct_old.txt";      % 36
    
    
    "particle_uFluct.txt";      % 37
    "particle_vFluct.txt";      % 38
    "particle_wFluct.txt";      % 39
    "particle_delta_uFluct.txt";      % 40
    "particle_delta_vFluct.txt";      % 41
    "particle_delta_wFluct.txt";      % 42
    
    "particle_isActive.txt";      % 43
    
    "particle_xPos.txt";      % 44
    "particle_yPos.txt";      % 45
    "particle_zPos.txt";      % 46
    
    ];

%%% now stuff all those input information into the codeInputFiles variable
codeInputFiles = cell(2,1);
codeInputFiles(1) = {codeInputDirs};
codeInputFiles(2) = {folderFileNames};


%%% set the plotOutputDir
plotOutputDir = sprintf("%s/../../d_plotOutput/b_channel/e_resultPlots",workingDir);



%%% now do some coding, using each of the specific functions to generate
%%% whatever plot you want


% now load in data
[fileExists_array, saveBasename_array,  current_time_array,timestep_array,  C_0_array,nParticles_array,  xCellGrid_array,yCellGrid_array,zCellGrid_array,  uMean_data_array,vMean_data_array,wMean_data_array,sigma2_data_array,epps_data_array,txx_data_array,txy_data_array,txz_data_array,tyy_data_array,tyz_data_array,tzz_data_array,  dtxxdx_data_array,dtxydx_data_array,dtxzdx_data_array,dtxydy_data_array,dtyydy_data_array,dtyzdy_data_array,dtxzdz_data_array,dtyzdz_data_array,dtzzdz_data_array,  flux_div_x_data_array,flux_div_y_data_array,flux_div_z_data_array,  txx_old_array,txy_old_array,txz_old_array,tyy_old_array,tyz_old_array,tzz_old_array,uFluct_old_array,vFluct_old_array,wFluct_old_array,  uFluct_array,vFluct_array,wFluct_array,delta_uFluct_array,delta_vFluct_array,delta_wFluct_array,  rogueCount_array,isActive_array,  xPos_array,yPos_array,zPos_array] = loadCodeOutput(codeInputFiles);

% take the saveBasename_array and create the plotBasename_array
[plotBasename_array] = saveToPlotBasename(fileExists_array,saveBasename_array);



% now plot fig12 stuff
% this is heterogeneous anisotropic positions
% is repeating the plots for the LES case for this case
%%% set the index of overall list of values paths for each plot of interest
fig12plotOutputIndices = [12,11,10, 15,14,13, 18,17,16];
%%% set figure plot filenames
fig12filename = sprintf("%s/channel_briansFig12Plot.png",plotOutputDir);
%%% set other needed values
nSubPlotRows = 3;
nSubPlotCols = 3;
subPlotIndices = 1:9;
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
fig15plotOutputIndices = [16,17,18];
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
expectedValueIdx = 10;
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
fig13plotOutputIndices = [16,17,18];
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
expectedValueIdx = 10;
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

