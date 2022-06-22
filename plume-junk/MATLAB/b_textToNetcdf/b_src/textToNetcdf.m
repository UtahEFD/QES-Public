function textToNetcdf(outputDir,outputBaseName,  x_turb_domainStart,x_turb_domainEnd,y_turb_domainStart,y_turb_domainEnd,z_turb_domainStart,z_turb_domainEnd, x_turb_nCells,y_turb_nCells,z_turb_nCells, x_turb_BCtype,y_turb_BCtype,z_turb_BCtype, C0,epps_file,sigma2_file,txx_file,txy_file,txz_file,tyy_file,tyz_file,tzz_file,  x_urb_domainStart,x_urb_domainEnd,y_urb_domainStart,y_urb_domainEnd,z_urb_domainStart,z_urb_domainEnd, x_urb_nCells,y_urb_nCells,z_urb_nCells, x_urb_BCtype,y_urb_BCtype,z_urb_BCtype, uMean_file,vMean_file,wMean_file)
    
    % the plan is to read in the data from each file into a data structure,
    % then manipulate it as needed to then output all the data structures
    % together into a netcdf file
    
    
    % if the input BCtypes are "", replace them with default values
    if x_turb_BCtype == ""
        x_turb_BCtype = "periodic";
    end
    if y_turb_BCtype == ""
        y_turb_BCtype = "periodic";
    end
    if z_turb_BCtype == ""
        z_turb_BCtype = "periodic";
    end
    if x_urb_BCtype == ""
        x_urb_BCtype = "periodic";
    end
    if y_urb_BCtype == ""
        y_urb_BCtype = "periodic";
    end
    if z_urb_BCtype == ""
        z_urb_BCtype = "periodic";
    end
    
    
    % the turb data is held differently than the urb data
    epps = readLinearized3DdataTextFile(epps_file,x_turb_nCells,y_turb_nCells,z_turb_nCells);
    sigma2 = readLinearized3DdataTextFile(sigma2_file,x_turb_nCells,y_turb_nCells,z_turb_nCells);
    txx = readLinearized3DdataTextFile(txx_file,x_turb_nCells,y_turb_nCells,z_turb_nCells);
    txy = readLinearized3DdataTextFile(txy_file,x_turb_nCells,y_turb_nCells,z_turb_nCells);
    txz = readLinearized3DdataTextFile(txz_file,x_turb_nCells,y_turb_nCells,z_turb_nCells);
    tyy = readLinearized3DdataTextFile(tyy_file,x_turb_nCells,y_turb_nCells,z_turb_nCells);
    tyz = readLinearized3DdataTextFile(tyz_file,x_turb_nCells,y_turb_nCells,z_turb_nCells);
    tzz = readLinearized3DdataTextFile(tzz_file,x_turb_nCells,y_turb_nCells,z_turb_nCells);
    
    % now create the turb position grid values
    % LA: turns out Bailey's code never uses the xCellGrid directly, just
    % the dx values. This means that for the periodic directions (the x and y
    % dimensions), the grid actually is like going from 0 to 1, but with
    % one extra cell, where the last cell is the same thing as the first
    % cell. I could change it now, and just do the grids to one step short,
    % but then particles would be confused about where the domain is
    % actually at. At least these current grids, even if not correct, have
    % been working pretty well in the past matlab code.
    %if x_turb_BCtype == "periodic"
    %    dx_turb = (x_turb_domainEnd - x_turb_domainStart)/(x_turb_nCells);
    %else
        dx_turb = (x_turb_domainEnd - x_turb_domainStart)/(x_turb_nCells - 1);
    %end
    %if y_turb_BCtype == "periodic"
    %    dy_turb = (y_turb_domainEnd - y_turb_domainStart)/(y_turb_nCells);
    %else
        dy_turb = (y_turb_domainEnd - y_turb_domainStart)/(y_turb_nCells - 1);
    %end
    %if z_turb_BCtype == "periodic"
    %    dz_turb = (z_turb_domainEnd - z_turb_domainStart)/(z_turb_nCells);
    %else
        dz_turb = (z_turb_domainEnd - z_turb_domainStart)/(z_turb_nCells - 1);
    %end
    if x_turb_nCells == 1
        dx_turb = 1;
    end
    if y_turb_nCells == 1
        dy_turb = 1;
    end
    if z_turb_nCells == 1
        dz_turb = 1;
    end
    xCellGrid_turb = x_turb_domainStart:dx_turb:x_turb_domainEnd';
    yCellGrid_turb = y_turb_domainStart:dy_turb:y_turb_domainEnd';
    zCellGrid_turb = z_turb_domainStart:dz_turb:z_turb_domainEnd';
    
    % now check the turb datasets
    [epps,sigma2,txx,txy,txz,tyy,tyz,tzz] = checkDatasets_turb(xCellGrid_turb,yCellGrid_turb,zCellGrid_turb,x_turb_nCells,y_turb_nCells,z_turb_nCells,epps,sigma2,txx,txy,txz,tyy,tyz,tzz);
    
    
    % the urb data is held differently than the turb data
    uMean = readLinearized3DdataTextFile(uMean_file,x_urb_nCells,y_urb_nCells,z_urb_nCells);
    vMean = readLinearized3DdataTextFile(vMean_file,x_urb_nCells,y_urb_nCells,z_urb_nCells);
    wMean = readLinearized3DdataTextFile(wMean_file,x_urb_nCells,y_urb_nCells,z_urb_nCells);
    
    % now create the urb position grid values
    %if x_urb_BCtype == "periodic"
        dx_urb = (x_urb_domainEnd - x_urb_domainStart)/(x_urb_nCells - 1);
    %else
    %    dx_urb = (x_urb_domainEnd - x_urb_domainStart)/(x_urb_nCells);
    %end
    %if y_urb_BCtype == "periodic"
        dy_urb = (y_urb_domainEnd - y_urb_domainStart)/(y_urb_nCells - 1);
    %else
    %    dy_urb = (y_urb_domainEnd - y_urb_domainStart)/(y_urb_nCells);
    %end
    %if z_urb_BCtype == "periodic"
        dz_urb = (z_urb_domainEnd - z_urb_domainStart)/(z_urb_nCells - 1);
    %else
    %    dz_urb = (z_urb_domainEnd - z_urb_domainStart)/(z_urb_nCells);
    %end
    if x_urb_nCells == 1
        dx_urb = 1;
    end
    if y_urb_nCells == 1
        dy_urb = 1;
    end
    if z_urb_nCells == 1
        dz_urb = 1;
    end
    xCellGrid_urb = x_urb_domainStart:dx_urb:x_urb_domainEnd';
    yCellGrid_urb = y_urb_domainStart:dy_urb:y_urb_domainEnd';
    zCellGrid_urb = z_urb_domainStart:dz_urb:z_urb_domainEnd';
    
    % now check the urb datasets
    [uMean,vMean,wMean] = checkDatasets_urb(xCellGrid_urb,yCellGrid_urb,zCellGrid_urb,x_urb_nCells,y_urb_nCells,z_urb_nCells,uMean,vMean,wMean);
    
    
    
    %%% all the data should be processed now. It is time to write the
    %%% netcdf files
    
    % now save the netcdf turb output
    writeNetcdfFile_turb(outputDir,outputBaseName,  xCellGrid_turb,yCellGrid_turb,zCellGrid_turb, C0,epps,sigma2,txx,txy,txz,tyy,tyz,tzz);
    
    % now save the netcdf turb output
    writeNetcdfFile_urb(outputDir,outputBaseName,  xCellGrid_urb,yCellGrid_urb,zCellGrid_urb, uMean,vMean,wMean);
    
    
end