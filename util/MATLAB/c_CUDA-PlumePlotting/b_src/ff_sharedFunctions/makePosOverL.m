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
    
    
    % correct the cell grids to be posOverL if they aren't already 
    %  posOverL grids
    posOverL = posCellGrid;
    if min(posOverL) ~= 0
        % correct it to be starting at zero
        posOverL = posOverL - min(posOverL);
    end
    pos_nCells = length(posOverL);
    if max(posOverL) ~= 1
        % correct it to be posOverL
        posOverL = posOverL/max(posOverL);
    end
    
    
end