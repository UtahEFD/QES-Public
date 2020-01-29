

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
mainCodeDir = sprintf("%s/../../b_src/a_plotEulerianVals",workingDir);
sharedFunctionsDir = sprintf("%s/../ff_sharedFunctions",mainCodeDir);

specificFunctionsDir = sprintf("%s/functions",mainCodeDir);
addpath(mainCodeDir,sharedFunctionsDir,  specificFunctionsDir);


%%%%%%% end standard main function stuff


%% LES case


%%% set the input file directory (codeInputDir) as well as the file base names
baseCodeInputDir = sprintf("%s/../../../../../testCases/Bailey/c_LES/c_plumeOutputs",workingDir);
codeInputFolders = [
    
    "a_0o0222_222";         % 1
    "b_0o222_222";          % 2
    "c_2o22_222";           % 3
    "d_22o2_222";           % 4
    
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
plotOutputDir = sprintf("%s/../../../../../testCases/Bailey/c_LES/e_matlabPlotOutput/a_inputPlots",workingDir);

% set the outputPlotFilenames using the basenames and plotOutputDir
plotFiles = strings(nCodeInputFiles,1);
for fileIdx = 1:nCodeInputFiles
    current_plotFileName = strrep(codeInputFolders(fileIdx),'/','_');
    plotFiles(fileIdx) = sprintf("%s/%s",plotOutputDir,current_plotFileName);    % still need to add _input vs _initial and .png
end


%%% set additional plot variables necessary for the plots

hozAvg = true;  % this has to always be true for any datasets that are not one dimensional

% % nonDim = false;
% % uMeanLim = "";
% % sigma2Lim = "";
% % eppsLim = "";

nonDim = true;
uMeanLim = [0,30];
sigma2Lim = [0,5];
eppsLim = [0,100];

% now used if not nondimensionalizing the plot
del = 1000;
u_tao = 0.45045;


% now finally run the function
plotEulerianVals(codeInputFiles,plotFiles, hozAvg, nonDim, uMeanLim,sigma2Lim,eppsLim, u_tao,del);



