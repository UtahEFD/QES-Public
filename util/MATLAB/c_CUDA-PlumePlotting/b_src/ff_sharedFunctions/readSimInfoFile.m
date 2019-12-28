function [simVarsExist,  saveBasename,  C_0,timestep,  current_time,rogueCount,  x_nCells,y_nCells,z_nCells,nParticles] = readSimInfoFile(simInfoFile)
    
    %%% needed to turn this into a function because the files aren't always
    %%% guaranteed to be the exact same every time
    %%% a very few variables can be set if they aren't in the output, but
    %%% most of the variables need to be in the file
    %%% if anything goes wrong, set the simVarsExist variable to false and
    %%% return
    
    % set the simVarsExist variable to be true so it will always be true
    % unless it is set to false by errors
    simVarsExist = true;
    
    % first make sure the file exists
    if exist(simInfoFile, 'file') ~= 2
        simVarsExist = false;
        return;
    end
    
    
    % now try to open the file, and scan in the data
    fileID = fopen(simInfoFile);
    if fileID == -1
        simVarsExist = false;
        return;
    end
    
    %%% I'm not sure if this is the smartest way to read in this file
    %%% but the current method creates a 1x3 cell array
    %%% where the first column is the variable names
    %%% the second column is the equals signs
    %%% and the third column is the values for the variables
    %%% where the whitespaces are left out
    %%% I'm not sure how this method handles lines that are different
    %%% but it does seem to work when there are empty lines in between some
    %%% variables
    textContents_cellPacked = textscan(fileID,'%s%s%s');
    fclose(fileID);
    varNames = string(textContents_cellPacked{1});
    %equalSigns = string(textContents_cellPacked{2});    % not used
    varValues = string(textContents_cellPacked{3});
    
    
    nLines = length(varNames);
    
    
    %%% now need to pull out the variables, setting them, exiting with
    %%% error, or putting in values if they aren't in the file
    
    saveBasename = findVar("saveBasename", varNames,varValues,nLines);
    if ~isstring(saveBasename)
        error("!!! readSimInfoFile error !!! saveBasename variable not found in input simInfoFile!");
    end
    
    
    C_0 = findVar("C_0", varNames,varValues,nLines);
    if ~isstring(C_0)
        error("!!! readSimInfoFile error !!! C_0 variable not found in input simInfoFile!");
    end
    timestep = findVar("timestep", varNames,varValues,nLines);
    if ~isstring(timestep)
        error("!!! readSimInfoFile error !!! timestep variable not found in input simInfoFile!");
    end
    
    
    current_time = findVar("current_time", varNames,varValues,nLines);
    if ~isstring(current_time)
        error("!!! readSimInfoFile error !!! current_time variable not found in input simInfoFile!");
    end
    rogueCount = findVar("rogueCount", varNames,varValues,nLines);
    if ~isstring(rogueCount)
        error("!!! readSimInfoFile error !!! rogueCount variable not found in input simInfoFile!");
    end
    
    
    x_nCells = findVar("Nx", varNames,varValues,nLines);
    if ~isstring(x_nCells)
        error("!!! readSimInfoFile error !!! Nx variable not found in input simInfoFile!");
    end
    y_nCells = findVar("Ny", varNames,varValues,nLines);
    if ~isstring(y_nCells)
        error("!!! readSimInfoFile error !!! Ny variable not found in input simInfoFile!");
    end
    z_nCells = findVar("Nz", varNames,varValues,nLines);
    if ~isstring(z_nCells)
        error("!!! readSimInfoFile error !!! Nz variable not found in input simInfoFile!");
    end
    nParticles = findVar("nParticles", varNames,varValues,nLines);
    if ~isstring(nParticles)
        error("!!! readSimInfoFile error !!! nParticles variable not found in input simInfoFile!");
    end
    
    
    %%% now need to overwrite the variable values that are not supposed to
    %%% be strings, with their converted values
    
    C_0 = str2double(C_0);
    timestep = str2double(timestep);
    
    current_time = str2double(current_time);
    rogueCount = str2double(rogueCount); 
    
    x_nCells = str2double(x_nCells);
    y_nCells = str2double(y_nCells);
    z_nCells = str2double(z_nCells);
    nParticles = str2double(nParticles);
    
end

function varValue = findVar(desiredVarName, varNames,varValues,nLines)
    
    % if the varName isn't found, varValue is set to NaN, otherwise it is
    % given a value. By starting out with NaN, the value is only set if it
    % is found
    varValue = NaN;
    for lineIdx = 1:nLines
        if desiredVarName == varNames(lineIdx)
            varValue = varValues(lineIdx);
            break;
        end
    end
    
end