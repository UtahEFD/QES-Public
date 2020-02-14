function [doesSimExist,     saveBasename, C_0,timestep, current_time,rogueCount,isNotActiveCount, invarianceTol,velThreshold,    urb_times,urb_x,urb_y,urb_z, urb_u,urb_v,urb_w,turb_sig_x,turb_sig_y,turb_sig_z,turb_txx,turb_txy,turb_txz,turb_tyy,turb_tyz,turb_tzz,turb_epps,turb_tke,eul_dtxxdx,eul_dtxydy,eul_dtxzdz,eul_dtxydx,eul_dtyydy,eul_dtyzdz,eul_dtxzdx,eul_dtyzdy,eul_dtzzdz,eul_flux_div_x,eul_flux_div_y,eul_flux_div_z,   lagrToEul_times,lagrToEul_x,lagrToEul_y,lagrToEul_z, lagrToEul_conc,    lagr_times,lagr_parID,  lagr_xPos_init,lagr_yPos_init,lagr_zPos_init,lagr_tStrt,lagr_sourceIdx,  lagr_xPos,lagr_yPos,lagr_zPos,lagr_uFluct,lagr_vFluct,lagr_wFluct,lagr_delta_uFluct,lagr_delta_vFluct,lagr_delta_wFluct,lagr_isRogue,lagr_isActive] = loadSingleCaseData(codeInputDir,folderFileNames)

    

    % now set the expected variable names for each input file
    % needs to be the same number of elements as the folderFileNames
    expectedVarNames = [

        "sim_info.txt";      % 1

        "_eulerianData.nc";    % 2
        "_conc.nc";            % 3
        "_particleInfo.nc";    % 4

        ];
    nFolderFiles = length(expectedVarNames);
    
    
    % now validate that the inputs are of the expected type and size
    % constraints as described in the function description above
    if iscell(codeInputDir)
        error('!!! loadSingleCaseData error !!! input codeInputDir is a cell array!');
    end
    if iscell(folderFileNames)
        error('!!! loadSingleCaseData error !!! input folderFileNames is a cell array!');
    end
    
    
    if length(size(codeInputDir)) ~= 2
        error('!!! loadSingleCaseData error !!! input codeInputDir has more than two dimensions!');
    end
    if size(codeInputDir,1) ~= 1 && size(codeInputDir,2) ~= 1
        error('!!! loadSingleCaseData error !!! input codeInputDir is more than one value!');
    end
    
    if length(size(folderFileNames)) ~= 2
        error('!!! loadSingleCaseData error !!! input folderFileNames has more than two dimensions!');
    end
    if size(folderFileNames,1) ~= 1 && size(folderFileNames,2) ~= 1
        error('!!! loadSingleCaseData error !!! input folderFileNames is not a row or a column vector!');
    end
    if length(folderFileNames) ~= nFolderFiles
        error('!!! loadSingleCaseData error !!! input folderFileNames does not have length %d, the size of expectedVarNames! input folderFileNames has length %d!',nFolderFiles,length(folderFileNames));
    end
    
    
    %%% all the input sizes have been checked. Any more error checking
    %%% needs to be done in the moment
    
    
    %%% need to set all the outputs to NAN, that way they are overridden
    %%% with the desired value, but it still runs if it quits early
    
    saveBasename = NaN;
    
    C_0 = NaN;
    timestep = NaN;

    current_time = NaN;
    rogueCount = NaN;
    isNotActiveCount = NaN;
    
    invarianceTol = NaN;
    velThreshold = NaN;

    
    
    urb_t = NaN;
    urb_x = NaN;
    urb_y = NaN;
    urb_z = NaN;
    urb_u = NaN;
    urb_v = NaN;
    urb_w = NaN;
    turb_sig_x = NaN;
    turb_sig_y = NaN;
    turb_sig_z = NaN;
    turb_txx = NaN;
    turb_txy = NaN;
    turb_txz = NaN;
    turb_tyy = NaN;
    turb_tyz = NaN;
    turb_tzz = NaN;
    turb_epps = NaN;
    turb_tke = NaN;
    eul_dtxxdx = NaN;
    eul_dtxydy = NaN;
    eul_dtxzdz = NaN;
    eul_dtxydx = NaN;
    eul_dtyydy = NaN;
    eul_dtyzdz = NaN;
    eul_dtxzdx = NaN;
    eul_dtyzdy = NaN;
    eul_dtzzdz = NaN;
    eul_flux_div_x = NaN;
    eul_flux_div_y = NaN;
    eul_flux_div_z = NaN;
    
    
    
    lagrToEul_times = NaN;
    lagrToEul_x = NaN;
    lagrToEul_y = NaN;
    lagrToEul_z = NaN;
    lagrToEul_conc = NaN;
    
    
    
    lagr_times = NaN;
    lagr_parID = NaN;
    lagr_xPos_init = NaN;
    lagr_yPos_init = NaN;
    lagr_zPos_init = NaN;
    lagr_tStrt = NaN;
    lagr_sourceIdx = NaN;
    lagr_xPos = NaN;
    lagr_yPos = NaN;
    lagr_zPos = NaN;
    lagr_uFluct = NaN;
    lagr_vFluct = NaN;
    lagr_wFluct = NaN;
    lagr_delta_uFluct = NaN;
    lagr_delta_vFluct = NaN;
    lagr_delta_wFluct = NaN;
    lagr_isRogue = NaN;
    lagr_isActive = NaN;
    
    
    
    %%% now start doing the loading stuff
    
    
    doesSimExist = true;

    % now read in the required information from the simInfoFile
    simInfoFile = sprintf("%s/%s",codeInputDir,folderFileNames(1));
    [simVarsExist,  saveBasename,  C_0,timestep,  current_time,rogueCount,isNotActiveCount,  invarianceTol,velThreshold] = readSimInfoFile(simInfoFile);
    % verify the sim info exists
    if simVarsExist == false
        doesSimExist = false;
        return;
    end
    
    % now read in the required information from the Eulerian file
    eulFile = sprintf("%s/%s",codeInputDir,folderFileNames(2));
    [simVarsExist,  urb_times,urb_x,urb_y,urb_z, urb_u,urb_v,urb_w,turb_sig_x,turb_sig_y,turb_sig_z,turb_txx,turb_txy,turb_txz,turb_tyy,turb_tyz,turb_tzz,turb_epps,turb_tke,eul_dtxxdx,eul_dtxydy,eul_dtxzdz,eul_dtxydx,eul_dtyydy,eul_dtyzdz,eul_dtxzdx,eul_dtyzdy,eul_dtzzdz,eul_flux_div_x,eul_flux_div_y,eul_flux_div_z] = readNetcdfEulerianFile(eulFile);
    % verify the sim info exists
    if simVarsExist == false
        doesSimExist = false;
        return;
    end
    
    % now read in the required information from the LagrToEul concentration
    % file
    lagrToEulFile = sprintf("%s/%s",codeInputDir,folderFileNames(3));
    [simVarsExist,  lagrToEul_times,lagrToEul_x,lagrToEul_y,lagrToEul_z, lagrToEul_conc] = readNetcdfLagrToEulFile(lagrToEulFile);
    % verify the sim info exists
    if simVarsExist == false
        doesSimExist = false;
        return;
    end
    
    % now read in the required information from the Lagrangian particle
    % file
    lagrFile = sprintf("%s/%s",codeInputDir,folderFileNames(4));
    [simVarsExist,  lagr_times,lagr_parID,  lagr_xPos_init,lagr_yPos_init,lagr_zPos_init,lagr_tStrt,lagr_sourceIdx,  lagr_xPos,lagr_yPos,lagr_zPos,lagr_uFluct,lagr_vFluct,lagr_wFluct,lagr_delta_uFluct,lagr_delta_vFluct,lagr_delta_wFluct,lagr_isRogue,lagr_isActive] = readNetcdfLagrangianFile(lagrFile);
    % verify the sim info exists
    if simVarsExist == false
        doesSimExist = false;
        return;
    end    

end
