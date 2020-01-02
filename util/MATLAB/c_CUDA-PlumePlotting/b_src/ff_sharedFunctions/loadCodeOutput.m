function [fileExists_array, saveBasename_array,  current_time_array,timestep_array,  C_0_array,nParticles_array,  xCellGrid_array,yCellGrid_array,zCellGrid_array,  uMean_data_array,vMean_data_array,wMean_data_array,sigma2_data_array,epps_data_array,txx_data_array,txy_data_array,txz_data_array,tyy_data_array,tyz_data_array,tzz_data_array,  dtxxdx_data_array,dtxydx_data_array,dtxzdx_data_array,dtxydy_data_array,dtyydy_data_array,dtyzdy_data_array,dtxzdz_data_array,dtyzdz_data_array,dtzzdz_data_array,  flux_div_x_data_array,flux_div_y_data_array,flux_div_z_data_array,  txx_old_array,txy_old_array,txz_old_array,tyy_old_array,tyz_old_array,tzz_old_array,uFluct_old_array,vFluct_old_array,wFluct_old_array,  uFluct_array,vFluct_array,wFluct_array,delta_uFluct_array,delta_vFluct_array,delta_wFluct_array,  rogueCount_array,isActive_array,  xPos_array,yPos_array,zPos_array] = loadCodeOutput(codeInputFiles)
    
    %%% the input codeInputFiles is expected to have the following shape
    %%% 
    %%% a row or a column cell array with length three
    %%% where cell(1) holds a row or a column string array (not cell array) 
    %%%    of folders, where each folder represents a single simulation
    %%% where cell(2) holds a row or a column string array (not cell array)
    %%%    of expected filenames to be found in each folder
    %%%   the contents of cell(2) has to hold files for the variables in
    %%%   the specific order as given by the variable "expectedVarNames"
    %%%   given at the start of this code. There are so many variables and
    %%%   in one specific order, it is better to refer to this variable and
    %%%   the code in this file than to list it all here.
    %%%   the most important filename is the first one, 
    %%%    which has to be the sim_info.txt file, so variables like the
    %%%    nx, ny, nz, and nParticles can be extracted for use reading 
    %%%    all the other files
    %%%
    %%% note that a bunch of the variables are not included in any of the 
    %%% input files found in codeInputFiles so these variables are given
    %%% no data values of -9999 or values from the existing datasets as
    %%% needed
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

        "sim_info";      % 1

        "xCellGrid";    % 2
        "yCellGrid";    % 3
        "zCellGrid";    % 4

        
        "uMean_data";       % 5
        "vMean_data";       % 6
        "wMean_data";       % 7
        "sigma2_data";      % 8
        "epps_data";        % 9
        "txx_data";        % 10
        "txy_data";        % 11
        "txz_data";        % 12
        "tyy_data";        % 13
        "tyz_data";        % 14
        "tzz_data";        % 15

        "dtxxdx_data";      % 16
        "dtxydx_data";      % 17
        "dtxzdx_data";      % 18
        "dtxydy_data";      % 19
        "dtyydy_data";      % 20
        "dtyzdy_data";      % 21
        "dtxzdz_data";      % 22
        "dtyzdz_data";      % 23
        "dtzzdz_data";      % 24

        "flux_div_x_data";      % 25
        "flux_div_y_data";      % 26
        "flux_div_z_data";      % 27


        "txx_old";      % 28
        "txy_old";      % 29
        "txz_old";      % 30
        "tyy_old";      % 31
        "tyz_old";      % 32
        "tzz_old";      % 33
        "uFluct_old";      % 34
        "vFluct_old";      % 35
        "wFluct_old";      % 36

        
        "uFluct";      % 37
        "vFluct";      % 38
        "wFluct";      % 39
        "delta_uFluct";      % 40
        "delta_vFluct";      % 41
        "delta_wFluct";      % 42

        "isActive";      % 43

        "xPos";      % 44
        "yPos";      % 45
        "zPos";      % 46

        ];
    nFolderFiles = length(expectedVarNames);
    
    
    % set the noDataVal to be used if variables are to exist but be kept
    % empty
    noDataVal = -9999;
    
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
    
    txx_old_array = cell(nSims,1);
    txy_old_array = cell(nSims,1);
    txz_old_array = cell(nSims,1);
    tyy_old_array = cell(nSims,1);
    tyz_old_array = cell(nSims,1);
    tzz_old_array = cell(nSims,1);
    uFluct_old_array = cell(nSims,1);
    vFluct_old_array = cell(nSims,1);
    wFluct_old_array = cell(nSims,1);
    
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
        
        doesSimExist = false;
        currentFolder = codeInputDirs(fileIdx);
        
        
        % now read in the required information from the simInfoFile
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(1));
        [simVarsExist,  saveBasename,  C_0,timestep,  current_time,rogueCount,  x_nCells,y_nCells,z_nCells,nParticles] = readSimInfoFile(currentFileName);
        % verify the sim info exists
        if simVarsExist == false
            fileExists_array(fileIdx) = false;
            continue;
        else
            doesSimExist = true;
        end
        
        % go through each variable and either set stuff based off of simInfoFile
        % or read in the variable from the required data file
        % the idea is to use these variables to later stuff into the cell
        % arrays
        
        %saveBasename = saveBasename;
        % now replace all decimals in dataName with "o" characters
        saveBasename = strrep(saveBasename,'.','o');
        
        %current_time = current_time;
        %timestep = timestep;
        
        %C_0 = C_0;
        %nParticles = nParticles;
        
        
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(2));
        xCellGrid = read1DdataTextFile(currentFileName,x_nCells);
        if isnan(xCellGrid)
            fileExists_array(fileIdx) = false;
            continue;
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(3));
        yCellGrid = read1DdataTextFile(currentFileName,y_nCells);
        if isnan(yCellGrid)
            fileExists_array(fileIdx) = false;
            continue;
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(4));
        zCellGrid = read1DdataTextFile(currentFileName,z_nCells);
        if isnan(zCellGrid)
            fileExists_array(fileIdx) = false;
            continue;
        end
        
        
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(5));
        uMean_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(uMean_data)
            uMean_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(6));
        vMean_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(vMean_data)
            vMean_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(7));
        wMean_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(wMean_data)
            wMean_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(8));
        sigma2_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(sigma2_data)
            sigma2_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(9));
        epps_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(epps_data)
            epps_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(10));
        txx_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(txx_data)
            txx_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(11));
        txy_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(txy_data)
            txy_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(12));
        txz_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(txz_data)
            txz_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(13));
        tyy_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(tyy_data)
            tyy_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(14));
        tyz_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(tyz_data)
            tyz_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(15));
        tzz_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(tzz_data)
            tzz_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(16));
        dtxxdx_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(dtxxdx_data)
            dtxxdx_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(17));
        dtxydx_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(dtxydx_data)
            dtxydx_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(18));
        dtxzdx_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(dtxzdx_data)
            dtxzdx_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(19));
        dtxydy_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(dtxydy_data)
            dtxydy_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(20));
        dtyydy_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(dtyydy_data)
            dtyydy_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(21));
        dtyzdy_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(dtyzdy_data)
            dtyzdy_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(22));
        dtxzdz_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(dtxzdz_data)
            dtxzdz_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(23));
        dtyzdz_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(dtyzdz_data)
            dtyzdz_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(24));
        dtzzdz_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(dtzzdz_data)
            dtzzdz_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(25));
        flux_div_x_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(flux_div_x_data)
            flux_div_x_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(26));
        flux_div_y_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(flux_div_y_data)
            flux_div_y_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(27));
        flux_div_z_data = readLinearized3DdataTextFile(currentFileName,x_nCells,y_nCells,z_nCells);
        if isnan(flux_div_z_data)
            flux_div_z_data = noDataVal*ones(x_nCells,y_nCells,z_nCells);
        end
        
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(28));
        txx_old = read1DdataTextFile(currentFileName,nParticles);
        if isnan(txx_old)
            txx_old = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(29));
        txy_old = read1DdataTextFile(currentFileName,nParticles);
        if isnan(txy_old)
            txy_old = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(30));
        txz_old = read1DdataTextFile(currentFileName,nParticles);
        if isnan(txz_old)
            txz_old = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(31));
        tyy_old = read1DdataTextFile(currentFileName,nParticles);
        if isnan(tyy_old)
            tyy_old = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(32));
        tyz_old = read1DdataTextFile(currentFileName,nParticles);
        if isnan(tyz_old)
            tyz_old = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(33));
        tzz_old = read1DdataTextFile(currentFileName,nParticles);
        if isnan(tzz_old)
            tzz_old = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(34));
        uFluct_old = read1DdataTextFile(currentFileName,nParticles);
        if isnan(uFluct_old)
            uFluct_old = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(35));
        vFluct_old = read1DdataTextFile(currentFileName,nParticles);
        if isnan(vFluct_old)
            vFluct_old = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(36));
        wFluct_old = read1DdataTextFile(currentFileName,nParticles);
        if isnan(wFluct_old)
            wFluct_old = noDataVal*ones(1,nParticles);
        end
        
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(37));
        uFluct = read1DdataTextFile(currentFileName,nParticles);
        if isnan(uFluct)
            uFluct = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(38));
        vFluct = read1DdataTextFile(currentFileName,nParticles);
        if isnan(vFluct)
            vFluct = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(39));
        wFluct = read1DdataTextFile(currentFileName,nParticles);
        if isnan(wFluct)
            wFluct = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(40));
        delta_uFluct = read1DdataTextFile(currentFileName,nParticles);
        if isnan(delta_uFluct)
            delta_uFluct = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(41));
        delta_vFluct = read1DdataTextFile(currentFileName,nParticles);
        if isnan(delta_vFluct)
            delta_vFluct = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(42));
        delta_wFluct = read1DdataTextFile(currentFileName,nParticles);
        if isnan(delta_wFluct)
            delta_wFluct = noDataVal*ones(1,nParticles);
        end
        
        %rogueCount = rogueCount;
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(43));
        isActive = read1DdataTextFile(currentFileName,nParticles);
        if isnan(isActive)
            isActive = noDataVal*ones(1,nParticles);
        end
        
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(44));
        xPos = read1DdataTextFile(currentFileName,nParticles);
        if isnan(xPos)
            xPos = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(45));
        yPos = read1DdataTextFile(currentFileName,nParticles);
        if isnan(yPos)
            yPos = noDataVal*ones(1,nParticles);
        end
        currentFileName = sprintf("%s/%s",currentFolder,folderFileNames(46));
        zPos = read1DdataTextFile(currentFileName,nParticles);
        if isnan(zPos)
            zPos = noDataVal*ones(1,nParticles);
        end
        
        
        
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

        txx_old_array(fileIdx) = {txx_old};
        txy_old_array(fileIdx) = {txy_old};
        txz_old_array(fileIdx) = {txz_old};
        tyy_old_array(fileIdx) = {tyy_old};
        tyz_old_array(fileIdx) = {tyz_old};
        tzz_old_array(fileIdx) = {tzz_old};
        uFluct_old_array(fileIdx) = {uFluct_old};
        vFluct_old_array(fileIdx) = {vFluct_old};
        wFluct_old_array(fileIdx) = {wFluct_old};

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
        %%% now delete the variables used to fill the cell arrays
        clear   'saveBasename'   'current_time' 'timestep'   'C_0' 'nParticles'   'x_nCells' 'y_nCells' 'z_nCells'  'xCellGrid' 'yCellGrid' 'zCellGrid'   'uMean_data' 'vMean_data' 'wMean_data' 'sigma2_data' 'epps_data' 'txx_data' 'txy_data' 'txz_data' 'tyy_data' 'tyz_data' 'tzz_data'   'dtxxdx_data' 'dtxydx_data' 'dtxzdx_data' 'dtxydy_data' 'dtyydy_data' 'dtyzdy_data' 'dtxzdz_data' 'dtyzdz_data' 'dtzzdz_data'   'flux_div_x_data' 'flux_div_y_data' 'flux_div_z_data'   'txx_old' 'txy_old' 'txz_old' 'tyy_old' 'tyz_old' 'tzz_old' 'uFluct_old' 'vFluct_old' 'wFluct_old'   'uFluct' 'vFluct' 'wFluct' 'delta_uFluct' 'delta_vFluct' 'delta_wFluct'   'rogueCount' 'isActive'   'xPos' 'yPos' 'zPos';
        
        
        % verify the simulation exists
        if doesSimExist == false
            fileExists_array(fileIdx) = false;
        else
            fileExists_array(fileIdx) = true;
        end
        
    end     % for loop nSims
    
end


function [data] = readLinearized3DdataTextFile(filename,x_nCells,y_nCells,z_nCells)

    % need to make sure the nCells are positive and nonzero
    if x_nCells < 1
        error('!!! readDataFile error!!! input x_nCells is not greater than 0!');
    end
    if y_nCells < 1
        error('!!! readDataFile error!!! input y_nCells is not greater than 0!');
    end
    if z_nCells < 1
        error('!!! readDataFile error!!! input z_nCells is not greater than 0!');
    end
    
    % if filename is "", return NaN for the data so that checkDatasets can
    % figure out how to return zeros for said dataset
    if filename == ""
        data = NaN;
        return;
    end

    % need to make sure the input filename exists
    if exist(filename, 'file') ~= 2
        %error('!!!input filename \"%s\" does not exist or is not a valid .txt file!',filename);
        data = NaN;
        return;
    end
    
    % now try to open the file, and scan in the data
    fileID = fopen(filename);
    if fileID == -1
        %error('could not open \"%s\" file!',filename);
        data = NaN;
        return;
    end
    data_cellPacked = textscan(fileID,'%f');
    data_linearized = cell2mat(data_cellPacked);
    fclose(fileID);
    
    % now need to put the linearized data into the output data matrix
    data = zeros(x_nCells,y_nCells,z_nCells);
    for kk = 1:z_nCells
        for jj = 1:y_nCells
            for ii = 1:x_nCells
                % hope this is the right order, fortran seems to not even make it easy
                % to know what is going on
                data(ii,jj,kk) = data_linearized((kk-1)*y_nCells*x_nCells + (jj-1)*x_nCells + ii);
            end
        end
    end
    
end


function [data] = read1DdataTextFile(filename,nVals)

    % need to make sure the nCells are positive and nonzero
    if nVals < 1
        error('!!! readDataFile error!!! input nVals is not greater than 0!');
    end
    
    % if filename is "", return NaN for the data so that checkDatasets can
    % figure out how to return zeros for said dataset
    if filename == ""
        data = NaN;
        return;
    end

    % need to make sure the input filename exists
    if exist(filename, 'file') ~= 2
        warning('!!!input filename \"%s\" does not exist or is not a valid .txt file!',filename);
        data = NaN;
        return;
    end
    
    % now try to open the file, and scan in the data
    fileID = fopen(filename);
    if fileID == -1
        warning('could not open \"%s\" file!',filename);
        data = NaN;
        return;
    end
    data_cellPacked = textscan(fileID,'%f');
    dataVals = cell2mat(data_cellPacked);
    fclose(fileID);
    
    if length(dataVals) ~= nVals
        warning('issues with \"%s\" file!',filename);
        data = NaN;
        return;
    end
    
    % now need to put the linearized data into the output data matrix
    data = zeros(1,nVals);
    for parIdx = 1:nVals
        data(parIdx) = dataVals(parIdx);
    end
    
end