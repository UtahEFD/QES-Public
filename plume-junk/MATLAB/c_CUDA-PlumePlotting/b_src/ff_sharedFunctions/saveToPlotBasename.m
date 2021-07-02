function [plotBasename_array] = saveToPlotBasename(fileExists_array,saveBasename_array)

    if (size(fileExists_array,1) ~= 1 && size(fileExists_array,2) ~= 1) || length(size(fileExists_array)) > 2
        error('!!! saveToPlotBasename error !!! input fileExists_array is not a row or a column vector!');
    end
    
    if (size(saveBasename_array,1) ~= 1 && size(saveBasename_array,2) ~= 1) || length(size(saveBasename_array)) > 2
        error('!!! saveToPlotBasename error !!! input saveBasename_array is not a row or a column vector!');
    end
    
    if length(fileExists_array) ~= length(saveBasename_array)
        error('!!! saveToPlotBasename error !!! input fileExists_array does not have the same number of values as the input saveBasename_array !');
    end
    
    % take the saveBasename_array and create the plotBasename_array. This
    % involves replacing all _ chars with " " chars
    nFiles = length(fileExists_array);
    plotBasename_array = cell(nFiles,1);
    for fileIdx = 1:nFiles
        if fileExists_array(fileIdx) == true
            current_saveBasename = string(saveBasename_array(fileIdx));
            % now replace all "_" chars with " " chars
            current_plotBasename = strrep(current_saveBasename,'_',' ');
            plotBasename_array(fileIdx) = {current_plotBasename};
        end
    end

end