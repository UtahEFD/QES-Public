function [epps,sigma2,txx,txy,txz,tyy,tyz,tzz] = checkDatasets_turb(xCellGrid,yCellGrid,zCellGrid,x_nCells,y_nCells,z_nCells,epps,sigma2,txx,txy,txz,tyy,tyz,tzz)

    % first make sure that nCells are all single values
    if length(size(x_nCells)) ~= 2
        error('!!! checkDatasets error !!! input x_nCells does not have 2 dimensions!');
    end
    if ~(size(x_nCells,1) == 1 && size(x_nCells,2) == 1)
        error('!!! checkDatasets error !!! input x_nCells is not a single value!');
    end
    if length(size(y_nCells)) ~= 2
        error('!!! checkDatasets error !!! input y_nCells does not have 2 dimensions!');
    end
    if ~(size(y_nCells,1) == 1 && size(y_nCells,2) == 1)
        error('!!! checkDatasets error !!! input y_nCells is not a single value!');
    end
    if length(size(z_nCells)) ~= 2
        error('!!! checkDatasets error !!! input z_nCells does not have 2 dimensions!');
    end
    if ~(size(z_nCells,1) == 1 && size(z_nCells,2) == 1)
        error('!!! checkDatasets error !!! input z_nCells is not a single value!');
    end
    
    % now make sure that all nCell values are 1 or greater
    if x_nCells < 1
        error('!!! checkDatasets error!!! input x_nCells is not greater than 0!');
    end
    if y_nCells < 1
        error('!!! checkDatasets error!!! input y_nCells is not greater than 0!');
    end
    if z_nCells < 1
        error('!!! checkDatasets error!!! input z_nCells is not greater than 0!');
    end
    
    
    % now, if any of the input datasets are a single NaN value, need to
    % turn them into a vector of zeros with the sizes of nCells
    %%%% yeah so this is a special thing that makes a lot of other stuff
    %%%% work
    if isnan(epps(1,1))
        epps = zeros(x_nCells,y_nCells,z_nCells);
    end
    if isnan(sigma2(1,1))
        sigma2 = zeros(x_nCells,y_nCells,z_nCells);
    end
    if isnan(txx(1,1))
        txx = zeros(x_nCells,y_nCells,z_nCells);
    end
    if isnan(txy(1,1))
        txy = zeros(x_nCells,y_nCells,z_nCells);
    end
    if isnan(txz(1,1))
        txz = zeros(x_nCells,y_nCells,z_nCells);
    end
    if isnan(tyy(1,1))
        tyy = zeros(x_nCells,y_nCells,z_nCells);
    end
    if isnan(tyz(1,1))
        tyz = zeros(x_nCells,y_nCells,z_nCells);
    end
    if isnan(tzz(1,1))
        tzz = zeros(x_nCells,y_nCells,z_nCells);
    end
    
    
    % now make sure all datasets have the correct size
    if length(size(xCellGrid)) ~= 2
        error('!!! checkDatasets error !!! input xCellGrid does not have 2 dimensions!');
    end
    if size(xCellGrid,1) ~= 1 && size(xCellGrid,2) ~= 1
        error('!!! checkDatasets error !!! input xCellGrid is not a row or a column vector!');
    end
    if length(xCellGrid) ~= x_nCells
        error('!!! checkDatasets error !!! input xCellGrid does not have the same number of elements as x_nCells!');
    end
    
    if length(size(yCellGrid)) ~= 2
        error('!!! checkDatasets error !!! input yCellGrid does not have 2 dimensions!');
    end
    if size(yCellGrid,1) ~= 1 && size(yCellGrid,2) ~= 1
        error('!!! checkDatasets error !!! input yCellGrid is not a row or a column vector!');
    end
    if length(yCellGrid) ~= y_nCells
        error('!!! checkDatasets error !!! input yCellGrid does not have the same number of elements as y_nCells!');
    end
    
    if length(size(zCellGrid)) ~= 2
        error('!!! checkDatasets error !!! input zCellGrid does not have 2 dimensions!');
    end
    if size(zCellGrid,1) ~= 1 && size(zCellGrid,2) ~= 1
        error('!!! checkDatasets error !!! input zCellGrid is not a row or a column vector!');
    end
    if length(zCellGrid) ~= z_nCells
        error('!!! checkDatasets error !!! input zCellGrid does not have the same number of elements as z_nCells!');
    end
    
    
    if length(size(epps)) ~= 3
        error('!!! checkDatasets error !!! input epps does not have 3 dimensions!');
    end
    if size(epps,1) ~= x_nCells
        error('!!! checkDatasets error !!! input epps dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(epps,2) ~= y_nCells
        error('!!! checkDatasets error !!! input epps dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(epps,3) ~= z_nCells
        error('!!! checkDatasets error !!! input epps dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    if length(size(sigma2)) ~= 3
        error('!!! checkDatasets error !!! input sigma2 does not have 3 dimensions!');
    end
    if size(sigma2,1) ~= x_nCells
        error('!!! checkDatasets error !!! input sigma2 dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(sigma2,2) ~= y_nCells
        error('!!! checkDatasets error !!! input sigma2 dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(sigma2,3) ~= z_nCells
        error('!!! checkDatasets error !!! input sigma2 dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    if length(size(txx)) ~= 3
        error('!!! checkDatasets error !!! input txx does not have 3 dimensions!');
    end
    if size(txx,1) ~= x_nCells
        error('!!! checkDatasets error !!! input txx dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(txx,2) ~= y_nCells
        error('!!! checkDatasets error !!! input txx dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(txx,3) ~= z_nCells
        error('!!! checkDatasets error !!! input txx dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    if length(size(txy)) ~= 3
        error('!!! checkDatasets error !!! input txy does not have 3 dimensions!');
    end
    if size(txy,1) ~= x_nCells
        error('!!! checkDatasets error !!! input txy dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(txy,2) ~= y_nCells
        error('!!! checkDatasets error !!! input txy dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(txy,3) ~= z_nCells
        error('!!! checkDatasets error !!! input txy dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    if length(size(txz)) ~= 3
        error('!!! checkDatasets error !!! input txz does not have 3 dimensions!');
    end
    if size(txz,1) ~= x_nCells
        error('!!! checkDatasets error !!! input txz dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(txz,2) ~= y_nCells
        error('!!! checkDatasets error !!! input txz dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(txz,3) ~= z_nCells
        error('!!! checkDatasets error !!! input txz dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    if length(size(tyy)) ~= 3
        error('!!! checkDatasets error !!! input tyy does not have 3 dimensions!');
    end
    if size(tyy,1) ~= x_nCells
        error('!!! checkDatasets error !!! input tyy dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(tyy,2) ~= y_nCells
        error('!!! checkDatasets error !!! input tyy dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(tyy,3) ~= z_nCells
        error('!!! checkDatasets error !!! input tyy dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    if length(size(tyz)) ~= 3
        error('!!! checkDatasets error !!! input tyz does not have 3 dimensions!');
    end
    if size(tyz,1) ~= x_nCells
        error('!!! checkDatasets error !!! input tyz dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(tyz,2) ~= y_nCells
        error('!!! checkDatasets error !!! input tyz dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(tyz,3) ~= z_nCells
        error('!!! checkDatasets error !!! input tyz dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    if length(size(tzz)) ~= 3
        error('!!! checkDatasets error !!! input tzz does not have 3 dimensions!');
    end
    if size(tzz,1) ~= x_nCells
        error('!!! checkDatasets error !!! input tzz dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(tzz,2) ~= y_nCells
        error('!!! checkDatasets error !!! input tzz dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(tzz,3) ~= z_nCells
        error('!!! checkDatasets error !!! input tzz dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    
end