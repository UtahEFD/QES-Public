function [posOverL,pos_nCells] = makePosOverL(posCellGrid)

    % the idea here is to correct the cell grid to be posOverL in form. If
    %  the input cell grid is already posOverL in form, the output cell grid
    %  should be the same as the input cell grid
    
    
    % check that the inputs make sense
    if length(size(posCellGrid)) > 2
        error('!!! makePosOverL error !!! input posCellGrid has more than two dimensions!');
    end
    if size(posCellGrid,1) ~= 1 && size(posCellGrid,2) ~= 1
        error('!!! makePosOverL error !!! input posCellGrid is not a row or a column vector!');
    end
    
    % have to correct for the isRogue or not isActive particles
    % these will have a position value of -999 or some big negative number
    % this will throw off the grid correction! Probably need a better fix
    % later
    isActive_indices = find(posCellGrid ~= -999.0);
    if isempty(isActive_indices)
        error('!!! makePosOverL error !!! input posCellGrid has no positions that are considered active!');
    end
    active_posCellGrid = posCellGrid(isActive_indices);
    
    % correct the cell grids to be posOverL if they aren't already 
    %  posOverL grids. Watch out for isRogue or not isActive problems
    posOverL = posCellGrid;
    if min(active_posCellGrid) ~= 0
        % correct it to be starting at zero
        active_posCellGrid = active_posCellGrid - min(active_posCellGrid);
        posOverL(isActive_indices) = active_posCellGrid;
    end
    pos_nCells = length(posOverL);
    if max(active_posCellGrid) ~= 1
        % correct it to be posOverL
        active_posCellGrid = active_posCellGrid/max(active_posCellGrid);
        posOverL(isActive_indices) = active_posCellGrid;
    end
    
    
end