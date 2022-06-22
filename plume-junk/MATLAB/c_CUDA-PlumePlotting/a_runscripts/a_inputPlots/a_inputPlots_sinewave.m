

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


%% sinewave case


%%% set the input file directory (codeInputDir) as well as the file base names
caseBaseName = "sinewave";
baseCodeInputDir = sprintf("%s/../../../../../testCases/Bailey/a_sinewave/c_plumeOutputs",workingDir);
codeInputFolders = [
    
    "a_0o01_10";      % 1
    "b_0o05_10";      % 2
    "c_0o1_10";       % 3
    "d_4_10";         % 4
    
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
plotOutputDir = sprintf("%s/../../../../../testCases/Bailey/a_sinewave/e_matlabPlotOutput/a_inputPlots",workingDir);

% set the outputPlotFilenames using the basenames and plotOutputDir
plotFiles = strings(nCodeInputFiles,1);
for fileIdx = 1:nCodeInputFiles
    plotFiles(fileIdx) = sprintf("%s/%s",plotOutputDir,codeInputFolders(fileIdx));    % still need to add _input vs _initial and .png
end


%%% set additional plot variables necessary for the plots

hozAvg = false;

nonDim = false;
uMeanLim = "";
sigma2Lim = [0 3];
eppsLim = [0 4];

% now used if not nondimensionalizing the plot
u_tao = "";
del = "";


% now finally run the function
plotEulerianVals(codeInputFiles,plotFiles, hozAvg, nonDim, uMeanLim,sigma2Lim,eppsLim, u_tao,del);



