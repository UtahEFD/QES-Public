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
        error('!!!input filename \"%s\" does not exist or is not a valid .txt file!',filename);
    end
    
    % now try to open the file, and scan in the data
    fileID = fopen(filename);
    if fileID == -1
        error('could not open \"%s\" file!',filename);
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