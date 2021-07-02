function [uMean,vMean,wMean] = checkDatasets_urb(xCellGrid,yCellGrid,zCellGrid,x_nCells,y_nCells,z_nCells,uMean,vMean,wMean)

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
    if isnan(uMean(1,1))
        uMean = zeros(x_nCells,y_nCells,z_nCells);
    end
    if isnan(vMean(1,1))
        vMean = zeros(x_nCells,y_nCells,z_nCells);
    end
    if isnan(wMean(1,1))
        wMean = zeros(x_nCells,y_nCells,z_nCells);
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
    
    
    if length(size(uMean)) ~= 3
        error('!!! checkDatasets error !!! input uMean does not have 3 dimensions!');
    end
    if size(uMean,1) ~= x_nCells
        error('!!! checkDatasets error !!! input uMean dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(uMean,2) ~= y_nCells
        error('!!! checkDatasets error !!! input uMean dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(uMean,3) ~= z_nCells
        error('!!! checkDatasets error !!! input uMean dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    if length(size(vMean)) ~= 3
        error('!!! checkDatasets error !!! input vMean does not have 3 dimensions!');
    end
    if size(vMean,1) ~= x_nCells
        error('!!! checkDatasets error !!! input vMean dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(vMean,2) ~= y_nCells
        error('!!! checkDatasets error !!! input vMean dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(vMean,3) ~= z_nCells
        error('!!! checkDatasets error !!! input vMean dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    if length(size(wMean)) ~= 3
        error('!!! checkDatasets error !!! input wMean does not have 3 dimensions!');
    end
    if size(wMean,1) ~= x_nCells
        error('!!! checkDatasets error !!! input wMean dimension 1 does not have the same number of values as specified by input x_nCells!');
    end
    if size(wMean,2) ~= y_nCells
        error('!!! checkDatasets error !!! input wMean dimension 2 does not have the same number of values as specified by input y_nCells!');
    end
    if size(wMean,3) ~= z_nCells
        error('!!! checkDatasets error !!! input wMean dimension 3 does not have the same number of values as specified by input z_nCells!');
    end
    
    
end