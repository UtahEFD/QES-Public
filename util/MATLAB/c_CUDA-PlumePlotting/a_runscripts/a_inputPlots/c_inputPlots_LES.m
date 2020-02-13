

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
caseBaseName = "LES";
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
    
    sprintf("%s_eulerianData.nc",caseBaseName);    % 2
    sprintf("%s_conc.nc",caseBaseName);            % 3
    sprintf("%s_particleInfo.nc",caseBaseName);    % 4
    
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

%%% always run the default without nondimensionalizing

hozAvg = true;  % this has to always be true for any datasets that are not one dimensional

nonDim = false;
uMeanLim = "";
sigma2Lim = "";
eppsLim = "";

% now used if not nondimensionalizing the plot
del = 1000;
u_tao = 0.45045;


% now finally run the function
plotEulerianVals(codeInputFiles,plotFiles, hozAvg, nonDim, uMeanLim,sigma2Lim,eppsLim, u_tao,del);


%%% now run it again, but this time dimensionalizing
nonDim = true;
uMeanLim = [0,30];
sigma2Lim = [0,5];
eppsLim = [0,100];
plotEulerianVals(codeInputFiles,plotFiles, hozAvg, nonDim, uMeanLim,sigma2Lim,eppsLim, u_tao,del);



