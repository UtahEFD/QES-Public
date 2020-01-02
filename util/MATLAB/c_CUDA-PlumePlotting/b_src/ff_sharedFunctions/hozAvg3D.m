function [out_posOverL,outData] = hozAvg3D(inData,averagingDir,posCellGrid,x_nCells,y_nCells,z_nCells)

    % the idea here is that the input data averaged by taking the mean of
    % the mean of values over a given dimension. This expects a structured
    % grid, so it should only be run on the plume Eulerian datasets.
    
    % I originally did this with matlab builtin functions, then set it up
    % to work for different directions in case that was ever needed later.
    % At the time I thought it was needed, but it turned out that it was
    % not.
    
    % make sure averagingDir is "x","y",or "z"
    if averagingDir ~= "x" && averagingDir ~= "y" && averagingDir ~= "z"
        error('!!! hozAvg3D error !!! input averagingDir does not have a value of "x", "y", or "z"! The value is \"%s\"',averagingDir);
    end
    
    if length(size(posCellGrid)) > 2
        error('!!! hozAvg3D error !!! input posCellGrid has more than two dimensions!');
    end
    if size(posCellGrid,1) ~= 1 && size(posCellGrid,2) ~= 1
        error('!!! hozAvg3D error !!! input posCellGrid is not a row or a column vector!');
    end
    
    
    % this is still checking the inputs a bit, but also setting the output sizes
    if averagingDir == "x"
        if length(posCellGrid) ~= x_nCells
            error('!!! hozAvg3D error !!! input posCellGrid does not have length x_nCells and input averagingDir is x!');
        end
        outData = zeros(x_nCells,1);
    elseif averagingDir == "y"
        if length(posCellGrid) ~= y_nCells
            error('!!! hozAvg3D error !!! input posCellGrid does not have length y_nCells and input averagingDir is y!');
        end
        outData = zeros(y_nCells,1);
    else    % if averagingDir == "z"
        if length(posCellGrid) ~= z_nCells
            error('!!! hozAvg3D error !!! input posCellGrid does not have length z_nCells and input averagingDir is z!');
        end
        outData = zeros(z_nCells,1);
    end
    
    
    % correct the cell grid to be posOverL if it isn't already posOverL in form
    [out_posOverL,~] = makePosOverL(posCellGrid);
    
        
    % now average over each bin
    if averagingDir == "z"
        for kk = 1:z_nCells
            y_sum = 0;
            for jj = 1:y_nCells
                x_sum = 0;
                for ii = 1:x_nCells
                    x_sum = x_sum + inData(ii,jj,kk);
                end
                y_sum = y_sum + x_sum/x_nCells;
            end
            outData(kk) = y_sum/y_nCells;
        end
    elseif averagingDir == "y"
        for jj = 1:y_nCells
            z_sum = 0;
            for kk = 1:z_nCells
                x_sum = 0;
                for ii = 1:x_nCells
                    x_sum = x_sum + inData(ii,jj,kk);
                end
                z_sum = z_sum + x_sum/x_nCells;
            end
            outData(jj) = z_sum/z_nCells;
        end
    else    % averagingDir == "x"
        for ii = 1:x_nCells
            y_sum = 0;
            for jj = 1:y_nCells
                z_sum = 0;
                for kk = 1:z_nCells
                    z_sum = z_sum + inData(ii,jj,kk);
                end
                y_sum = y_sum + z_sum/z_nCells;
            end
            outData(ii) = y_sum/y_nCells;
        end
    end
    
    % the above does exactly the same thing as the first if condition set 
    % above, as far as averaging over the x first, then the y, then the z
    %outData = zeros(z_nCells,1,1);
    %outData(:) = mean(mean(inData));
    
end
