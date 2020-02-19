function [fileExists_array, saveBasename_array,  current_time_array,timestep_array,  C_0_array,nParticles_array,  xCellGrid_array,yCellGrid_array,zCellGrid_array,  uMean_data_array,vMean_data_array,wMean_data_array,sigma2_data_array,epps_data_array,txx_data_array,txy_data_array,txz_data_array,tyy_data_array,tyz_data_array,tzz_data_array,  dtxxdx_data_array,dtxydx_data_array,dtxzdx_data_array,dtxydy_data_array,dtyydy_data_array,dtyzdy_data_array,dtxzdz_data_array,dtyzdz_data_array,dtzzdz_data_array,  flux_div_x_data_array,flux_div_y_data_array,flux_div_z_data_array,  uFluct_array,vFluct_array,wFluct_array,delta_uFluct_array,delta_vFluct_array,delta_wFluct_array,  rogueCount_array,isActive_array,  xPos_array,yPos_array,zPos_array] = loadCodeOutput(codeInputFiles)
    
    %%% the input codeInputFiles is expected to have the following shape
    %%% 
    %%% a row or a column cell array with length three
    %%% where cell(1) holds a row or a column string array (not cell array) 
    %%%    of folders, where each folder represents a single simulation
    %%% where cell(2) holds a row or a column string array (not cell array)
    %%%    of expected filenames to be found in each folder
    %%%   the contents of cell(2) has to hold files for the variables in
    %%%   the specific order as given by the variable "expectedVarNames"
    %%%   given at the start of this code.
    %%%   the most important filename is the first file, 
    %%%    which has to be the sim_info.txt file, so variables like the
    %%%    nx, ny, nz, and nParticles can be extracted for use reading 
    %%%    all the other files
    %%%   as the other files are netcdf now instead of separate text files,
    %%%   each has expected variables to be found in them.
    %%%
    %%% note that if any variable is not found for a given case, the
    %%% code moves on to the next case, setting a bool to know the case 
    %%% is unusable
    %%%
    %%% all variables are loaded into cell arrays for each simulation
    %%% these arrays are type cell and not numeric arrays, this function 
    %%% should happily throw them into the containers without problems. But
    %%% this means that the users of these containers have to be careful 
    %%% about how they unpack them because the stuff inside may be of 
    %%% differing sizes.
    %%% Probably should have a different set of cell arrays/array files 
    %%% for each test case ie sinewave, channel, LES to make it easier to
    %%% keep track of info. Also, hopefully the amount of memory and other
    %%% constraints won't be a problem since there are so many values
    %%% getting passed around now.
    
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
    if ~iscell(codeInputFiles)
        error('!!! loadCodeOutput error !!! input codeInputFiles is not a cell array!');
    end
    if length(size(codeInputFiles)) ~= 2
        error('!!! loadCodeOutput error !!! input codeInputFiles has more than two dimensions!');
    end
    if size(codeInputFiles,1) ~= 1 && size(codeInputFiles,2) ~= 1
        error('!!! loadCodeOutput error !!! input codeInputFiles is not a row or a column vector!');
    end
    if length(codeInputFiles) ~= 2
        error('!!! loadCodeOutput error !!! input codeInputFiles does not have length of two!');
    end
    
    codeInputDirs = codeInputFiles{1};
    folderFileNames = codeInputFiles{2};
    
    if length(size(codeInputDirs)) ~= 2
        error('!!! loadCodeOutput error !!! input codeInputFiles(1) codeInputDirs has more than two dimensions!');
    end
    if size(codeInputDirs,1) ~= 1 && size(codeInputDirs,2) ~= 1
        error('!!! loadCodeOutput error !!! input codeInputFiles(1) codeInputDirs is not a row or a column vector!');
    end
    nSims = length(codeInputDirs);
    
    if length(size(folderFileNames)) ~= 2
        error('!!! loadCodeOutput error !!! input codeInputFiles(2) folderFileNames has more than two dimensions!');
    end
    if size(folderFileNames,1) ~= 1 && size(folderFileNames,2) ~= 1
        error('!!! loadCodeOutput error !!! input codeInputFiles(2) folderFileNames is not a row or a column vector!');
    end
    if length(folderFileNames) ~= nFolderFiles
        error('!!! loadCodeOutput error !!! input codeInputFiles(2) folderFileNames does not have length %d, the size of expectedVarNames! input codeInputFiles(2) folderFileNames has length %d!',nFolderFiles,length(folderFileNames));
    end
    
    
    %%% all the input sizes have been checked. Any more error checking
    %%% needs to be done in the moment
    
    %%% start out by creating cell arrays of size (nSims,1)
    fileExists_array = true(nSims,1);
    
    saveBasename_array = cell(nSims,1);
    
    current_time_array = cell(nSims,1);
    timestep_array = cell(nSims,1);
    
    C_0_array = cell(nSims,1);
    nParticles_array = cell(nSims,1);
    
    xCellGrid_array = cell(nSims,1);
    yCellGrid_array = cell(nSims,1);
    zCellGrid_array = cell(nSims,1);
     
    uMean_data_array = cell(nSims,1);
    vMean_data_array = cell(nSims,1);
    wMean_data_array = cell(nSims,1);
    sigma2_data_array = cell(nSims,1);
    epps_data_array = cell(nSims,1);
    txx_data_array = cell(nSims,1);
    txy_data_array = cell(nSims,1);
    txz_data_array = cell(nSims,1);
    tyy_data_array = cell(nSims,1);
    tyz_data_array = cell(nSims,1);
    tzz_data_array = cell(nSims,1);
    
    dtxxdx_data_array = cell(nSims,1);
    dtxydx_data_array = cell(nSims,1);
    dtxzdx_data_array = cell(nSims,1);
    dtxydy_data_array = cell(nSims,1);
    dtyydy_data_array = cell(nSims,1);
    dtyzdy_data_array = cell(nSims,1);
    dtxzdz_data_array = cell(nSims,1);
    dtyzdz_data_array = cell(nSims,1);
    dtzzdz_data_array = cell(nSims,1);
    
    flux_div_x_data_array = cell(nSims,1);
    flux_div_y_data_array = cell(nSims,1);
    flux_div_z_data_array = cell(nSims,1);
    
    uFluct_array = cell(nSims,1);
    vFluct_array = cell(nSims,1);
    wFluct_array = cell(nSims,1);
    delta_uFluct_array = cell(nSims,1);
    delta_vFluct_array = cell(nSims,1);
    delta_wFluct_array = cell(nSims,1);
    
    rogueCount_array = cell(nSims,1);
    isActive_array = cell(nSims,1);
    
    xPos_array = cell(nSims,1);
    yPos_array = cell(nSims,1);
    zPos_array = cell(nSims,1);
    
    
    %%% now loop through each file filling it's values into the cell arrays
    %%% or setting values as needed that aren't found in the files
    for fileIdx = 1:nSims
        
        [doesSimExist,     saveBasename, C_0,timestep, current_time,rogueCount,isNotActiveCount,  invarianceTol,velThreshold,    urb_times,urb_x,urb_y,urb_z, urb_u,urb_v,urb_w,turb_sig_x,turb_sig_y,turb_sig_z,turb_txx,turb_txy,turb_txz,turb_tyy,turb_tyz,turb_tzz,turb_epps,turb_tke,eul_dtxxdx,eul_dtxydy,eul_dtxzdz,eul_dtxydx,eul_dtyydy,eul_dtyzdz,eul_dtxzdx,eul_dtyzdy,eul_dtzzdz,eul_flux_div_x,eul_flux_div_y,eul_flux_div_z,   lagrToEul_times,lagrToEul_x,lagrToEul_y,lagrToEul_z, lagrToEul_conc,    lagr_times,lagr_parID,  lagr_xPos_init,lagr_yPos_init,lagr_zPos_init,lagr_tStrt,lagr_sourceIdx,  lagr_xPos,lagr_yPos,lagr_zPos,lagr_uFluct,lagr_vFluct,lagr_wFluct,lagr_delta_uFluct,lagr_delta_vFluct,lagr_delta_wFluct,lagr_isRogue,lagr_isActive] = loadSingleCaseData(codeInputDirs(fileIdx),folderFileNames);
        
        % verify the simulation exists
        if doesSimExist == false
            fileExists_array(fileIdx) = false;
            continue;
        else
            fileExists_array(fileIdx) = true;
        end
        
        
        % now get the dimensions of the data
        nTimes = length(lagr_times);
        %nx = length(urb_x);
        %ny = length(urb_y);
        %nz = length(urb_z);
        nPar = length(lagr_parID);
        
        
        % now go through each variable to set the stuff that is required 
        % for the actual output
        % the idea is to use these variables to later stuff into the cell
        % arrays

        %saveBasename = saveBasename;
        % now replace all decimals in dataName with "o" characters
        saveBasename = strrep(saveBasename,'.','o');

        %current_time = current_time;
        %timestep = timestep;

        %C_0 = C_0;
        nParticles = nPar;


        xCellGrid = urb_x;
        yCellGrid = urb_y;
        zCellGrid = urb_z;

        uMean_data = urb_u;
        vMean_data = urb_v;
        wMean_data = urb_w;
        sigma2_data = turb_sig_z;
        epps_data = turb_epps;
        txx_data = turb_txx;
        txy_data = turb_txy;
        txz_data = turb_txz;
        tyy_data = turb_tyy;
        tyz_data = turb_tyz;
        tzz_data = turb_tzz;

        dtxxdx_data = eul_dtxxdx;
        dtxydx_data = eul_dtxydx;
        dtxzdx_data = eul_dtxzdx;
        dtxydy_data = eul_dtxydy;
        dtyydy_data = eul_dtyydy;
        dtyzdy_data = eul_dtyzdy;
        dtxzdz_data = eul_dtxzdz;
        dtyzdz_data = eul_dtyzdz;
        dtzzdz_data = eul_dtzzdz;

        flux_div_x_data = eul_flux_div_x;
        flux_div_y_data = eul_flux_div_y;
        flux_div_z_data = eul_flux_div_z;


        uFluct = lagr_uFluct(:,nTimes);
        vFluct = lagr_vFluct(:,nTimes);
        wFluct = lagr_wFluct(:,nTimes);
        delta_uFluct = lagr_delta_uFluct(:,nTimes);
        delta_vFluct = lagr_delta_vFluct(:,nTimes);
        delta_wFluct = lagr_delta_wFluct(:,nTimes);

        %rogueCount = rogueCount;
        isActive = lagr_isRogue(:,nTimes);

        xPos = lagr_xPos(:,nTimes);
        yPos = lagr_yPos(:,nTimes);
        zPos = lagr_zPos(:,nTimes);


        %%% there was a mistake in the output, somehow I set the isActive
        %%% to false if the particle was active, so need to reverse the
        %%% values here now.
        for valIdx = 1:length(isActive)
            if isActive(valIdx) == false
                isActive(valIdx) = true;
            elseif isActive(valIdx) == true
                isActive(valIdx) = false;
            end
        end
        
        
        
        
        %%% now all the values are set, start using them to fill the cell
        %%% arrays
        % now fill the values into the cell array
        saveBasename_array(fileIdx) = {saveBasename};

        current_time_array(fileIdx) = {current_time};
        timestep_array(fileIdx) = {timestep};

        C_0_array(fileIdx) = {C_0};
        nParticles_array(fileIdx) = {nParticles};

        xCellGrid_array(fileIdx) = {xCellGrid};
        yCellGrid_array(fileIdx) = {yCellGrid};
        zCellGrid_array(fileIdx) = {zCellGrid};

        uMean_data_array(fileIdx) = {uMean_data};
        vMean_data_array(fileIdx) = {vMean_data};
        wMean_data_array(fileIdx) = {wMean_data};
        sigma2_data_array(fileIdx) = {sigma2_data};
        epps_data_array(fileIdx) = {epps_data};
        txx_data_array(fileIdx) = {txx_data};
        txy_data_array(fileIdx) = {txy_data};
        txz_data_array(fileIdx) = {txz_data};
        tyy_data_array(fileIdx) = {tyy_data};
        tyz_data_array(fileIdx) = {tyz_data};
        tzz_data_array(fileIdx) = {tzz_data};

        dtxxdx_data_array(fileIdx) = {dtxxdx_data};
        dtxydx_data_array(fileIdx) = {dtxydx_data};
        dtxzdx_data_array(fileIdx) = {dtxzdx_data};
        dtxydy_data_array(fileIdx) = {dtxydy_data};
        dtyydy_data_array(fileIdx) = {dtyydy_data};
        dtyzdy_data_array(fileIdx) = {dtyzdy_data};
        dtxzdz_data_array(fileIdx) = {dtxzdz_data};
        dtyzdz_data_array(fileIdx) = {dtyzdz_data};
        dtzzdz_data_array(fileIdx) = {dtzzdz_data};

        flux_div_x_data_array(fileIdx) = {flux_div_x_data};
        flux_div_y_data_array(fileIdx) = {flux_div_y_data};
        flux_div_z_data_array(fileIdx) = {flux_div_z_data};

        uFluct_array(fileIdx) = {uFluct};
        vFluct_array(fileIdx) = {vFluct};
        wFluct_array(fileIdx) = {wFluct};
        delta_uFluct_array(fileIdx) = {delta_uFluct};
        delta_vFluct_array(fileIdx) = {delta_vFluct};
        delta_wFluct_array(fileIdx) = {delta_wFluct};

        rogueCount_array(fileIdx) = {rogueCount};
        isActive_array(fileIdx) = {isActive};

        xPos_array(fileIdx) = {xPos};
        yPos_array(fileIdx) = {yPos};
        zPos_array(fileIdx) = {zPos};

        % now delete the old loaded data just in case new loaded data doesn't
        % have it so avoiding that the old values are used on accident.
        %%% delete the variables created from loadSingleCaseData()
        %clear   'doesSimExist'   'saveBasename'  'C_0' 'timestep'  'current_time' 'rogueCount' 'isNotActiveCount'  'invarianceTol' 'velThreshold'     'urb_times' 'urb_x' 'urb_y' 'urb_z'  'urb_u' 'urb_v' 'urb_w' 'turb_sig_x' 'turb_sig_y' 'turb_sig_z' 'turb_txx' 'turb_txy' 'turb_txz' 'turb_tyy' 'turb_tyz' 'turb_tzz' 'turb_epps' 'turb_tke' 'eul_dtxxdx' 'eul_dtxydy' 'eul_dtxzdz' 'eul_dtxydx' 'eul_dtyydy' 'eul_dtyzdz' 'eul_dtxzdx' 'eul_dtyzdy' 'eul_dtzzdz' 'eul_flux_div_x' 'eul_flux_div_y' 'eul_flux_div_z'     'lagrToEul_times' 'lagrToEul_x' 'lagrToEul_y' 'lagrToEul_z'  'lagrToEul_conc'     'lagr_times' 'lagr_parID'  'lagr_xPos_init' 'lagr_yPos_init' 'lagr_zPos_init' 'lagr_tStrt' 'lagr_sourceIdx'  'lagr_xPos' 'lagr_yPos' 'lagr_zPos' 'lagr_uFluct' 'lagr_vFluct' 'lagr_wFluct' 'lagr_delta_uFluct' 'lagr_delta_vFluct' 'lagr_delta_wFluct' 'lagr_isRogue' 'lagr_isActive';
         clear   'doesSimExist'                                                                 'isNotActiveCount'  'invarianceTol' 'velThreshold'     'urb_times' 'urb_x' 'urb_y' 'urb_z'  'urb_u' 'urb_v' 'urb_w' 'turb_sig_x' 'turb_sig_y' 'turb_sig_z' 'turb_txx' 'turb_txy' 'turb_txz' 'turb_tyy' 'turb_tyz' 'turb_tzz' 'turb_epps' 'turb_tke' 'eul_dtxxdx' 'eul_dtxydy' 'eul_dtxzdz' 'eul_dtxydx' 'eul_dtyydy' 'eul_dtyzdz' 'eul_dtxzdx' 'eul_dtyzdy' 'eul_dtzzdz' 'eul_flux_div_x' 'eul_flux_div_y' 'eul_flux_div_z'     'lagrToEul_times' 'lagrToEul_x' 'lagrToEul_y' 'lagrToEul_z'  'lagrToEul_conc'     'lagr_times' 'lagr_parID'  'lagr_xPos_init' 'lagr_yPos_init' 'lagr_zPos_init' 'lagr_tStrt' 'lagr_sourceIdx'  'lagr_xPos' 'lagr_yPos' 'lagr_zPos' 'lagr_uFluct' 'lagr_vFluct' 'lagr_wFluct' 'lagr_delta_uFluct' 'lagr_delta_vFluct' 'lagr_delta_wFluct' 'lagr_isRogue' 'lagr_isActive';
        %%% now delete the variables used to fill the cell arrays
        clear   'saveBasename'   'current_time' 'timestep'   'C_0' 'nParticles'   'xCellGrid' 'yCellGrid' 'zCellGrid'   'uMean_data' 'vMean_data' 'wMean_data' 'sigma2_data' 'epps_data' 'txx_data' 'txy_data' 'txz_data' 'tyy_data' 'tyz_data' 'tzz_data'   'dtxxdx_data' 'dtxydx_data' 'dtxzdx_data' 'dtxydy_data' 'dtyydy_data' 'dtyzdy_data' 'dtxzdz_data' 'dtyzdz_data' 'dtzzdz_data'   'flux_div_x_data' 'flux_div_y_data' 'flux_div_z_data'   'uFluct' 'vFluct' 'wFluct' 'delta_uFluct' 'delta_vFluct' 'delta_wFluct'   'rogueCount' 'isActive'   'xPos' 'yPos' 'zPos';
        %%% delete any leftover intermediate variables
        clear   'nTimes' 'nPar';
        
    end     % for loop nSims
    
end
