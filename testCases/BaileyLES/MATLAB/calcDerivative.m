function [ dfdz ] = calcDerivative(input_data,dz)
    
    % this calculates the derivative field in the z direction for a 3D dataset
    % using 2nd order forward differencing for the first node,
    % central differencing for interior nodes,
    % and 2nd order backward differencing for the last node.
    % this function expects at least 5 values to work correctly without
    % error.
    
    % get the number of z values
    nz = size(input_data,3);
    
    % set the output container size
    dfdz = nan(size(input_data));
    
    % now calculate the derivative for the first node using 2nd order
    % forward differencing
    dfdz(:,:,1) = ( -3*input_data(:,:,1) + 4*input_data(:,:,2) - input_data(:,:,3) )/(2*dz);
    % now calculate the derivative for the interior nodes using central
    % differencing
    for kk = 2:nz-1
        dfdz(:,:,kk) = ( input_data(:,:,kk+1) - input_data(:,:,kk-1) )/(2*dz);
    end
    % now calculate the derivative for the last node using 2nd order
    % backward differencing
    dfdz(:,:,nz) = ( 3*input_data(:,:,nz) - 4*input_data(:,:,nz-1) + input_data(:,:,nz-2) )/(2*dz);
    
    
end