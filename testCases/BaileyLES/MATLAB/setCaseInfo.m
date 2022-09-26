function [ caseNames, timestep_array, currentTime_array ] = setCaseInfo(caseFolders,caseBaseName)
% this function is a bit quirky, because you could just specify the
% timestep array and the current time array and the case names as regular
% inputs in the main function kind of like how the case folders is set, but
% then you would need to make sure these variables stay in the same order
% and stay the same length as each other. Also, the case folders will probably
% always contain these variables in their names. Ideally, you would
% probably grab all these variables/names/values from some kind of input
% file for each separate simulation, but since they are already specified
% in the folder paths to get to said desired files (the .xml file), might
% as well use what we have. Probably would be better to just use the folder
% paths to read the input .xml files and the runscript files to get all 
% these variables and to group them all together, but for now this 
% is good enough.

    nCaseFolders = length(caseFolders);
    
    caseNames = strings(nCaseFolders,1);
    timestep_array = nan(nCaseFolders,1);
    currentTime_array = nan(nCaseFolders,1);
    
    for fileIdx = 1:nCaseFolders
        splitString = strsplit(caseFolders(fileIdx),"_");
        caseNames(fileIdx) = sprintf( "%s%s" , caseBaseName, strrep(  caseFolders(fileIdx),  splitString(1), ""  ) );
        timestep_array(fileIdx) = double( strrep(splitString(2),"o",".") );
        currentTime_array(fileIdx) = double( strrep(splitString(3),"o",".") );
    end
    
end