function [ outData ] = hozAvg3D( inData )

    % first get the number of values needed to size the output
    % notice that the data is expected to have 3 dimensions and nVals is
    % the number of z values there are in the dataset
    z_nVals = length(inData(1,1,:));
    
    
    % select the region of the inData that should be processed in the hozAvg
    % all z values will be processed, including the z domain edge ghost
    % nodes. only process the x and y values that are on or within the domain
    % edges, so filter out any x and y domain edge ghost nodes.
    % keeping all horizontally averaging regions within the domain avoids
    % adding in any averaging error (extra vals biases the average)
    %
    % because this function is only called for cell centered datasets for
    % these plots, the data only needs filtered for cell centered style
    % datasets. If this function is called for more complicated datasets,
    % see the version of this function used for the input plots for more
    % ideas.
    %
    % expected inData grid is ( x_cc, y_cc, z_cc ) or in other words
    % ( -dx/2:dx:Lx+dx/2, -dy/2:dy:Ly+dy/2, -dz/2:dz:Lz+dz/2 )
    % so need to ignore vals at ( -dx/2 and Lx+dx/2, -dy/2 and Ly+dy/2 )
    % keep all the z values
    processData = inData(2:end-1,2:end-1,:);
    
    
    % now size the output and horizontally average the data
    % this is the same thing as first averaging over all the x values for a given y value
    %  then averaging over all the y values for a given z value
    outData = zeros(z_nVals,1,1);
    outData(:) = mean(mean(processData));

end