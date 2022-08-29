function [ xGridInfo, yGridInfo, zGridInfo ] = calcGridInfo( x_cc, y_cc, z_cc )

    % first step is to make the x_fc, y_fc, and z_fc from the x_cc, y_cc,
    % and z_cc grids
    % the first cell should be a ghost cell, same with the last. There is
    % one extra value in the face centered values since these are edges
    % instead of cell centers.
    dx = x_cc(2) - x_cc(1);
    dy = y_cc(2) - y_cc(1);
    dz = z_cc(2) - z_cc(1);
    x_fc = [ x_cc(1)-dx/2; x_cc+dx/2 ];
    y_fc = [ y_cc(1)-dy/2; y_cc+dy/2 ];
    z_fc = [ z_cc(1)-dz/2; z_cc+dz/2 ];
    
    
    % now the number of values
    nx_cc = length(x_cc);
    ny_cc = length(y_cc);
    nz_cc = length(z_cc);
    nx_fc = length(x_fc);
    ny_fc = length(y_fc);
    nz_fc = length(z_fc);
    
    
    % now find Lx,Ly, and Lz. The ghost cells should be the first and last
    % cells in each direction, so Lx,Ly,Lz are one less than the last value
    Lx = x_fc(nx_fc-1);
    Ly = y_fc(ny_fc-1);
    Lz = z_fc(nz_fc-1);
    
    
    % now convert the x_cc, y_cc, z_cc, x_fc, y_fc, and z_fc grids to xOverL
    xOverL_cc = x_cc/Lx;
    yOverL_cc = y_cc/Ly;
    zOverL_cc = z_cc/Lz;
    xOverL_fc = x_fc/Lx;
    yOverL_fc = y_fc/Ly;
    zOverL_fc = z_fc/Lz;
    
    
    % now set the grid info structure variables
    xGridInfo.x_cc = x_cc;
    xGridInfo.x_fc = x_fc;
    xGridInfo.xOverL_cc = xOverL_cc;
    xGridInfo.xOverL_fc = xOverL_fc;
    xGridInfo.nx_cc = nx_cc;
    xGridInfo.nx_fc = nx_fc;
    xGridInfo.dx = dx;
    xGridInfo.Lx = Lx;
    
    yGridInfo.y_cc = y_cc;
    yGridInfo.y_fc = y_fc;
    yGridInfo.yOverL_cc = yOverL_cc;
    yGridInfo.yOverL_fc = yOverL_fc;
    yGridInfo.ny_cc = ny_cc;
    yGridInfo.ny_fc = ny_fc;
    yGridInfo.dy = dy;
    yGridInfo.Ly = Ly;
    
    zGridInfo.z_cc = z_cc;
    zGridInfo.z_fc = z_fc;
    zGridInfo.zOverL_cc = zOverL_cc;
    zGridInfo.zOverL_fc = zOverL_fc;
    zGridInfo.nz_cc = nz_cc;
    zGridInfo.nz_fc = nz_fc;
    zGridInfo.dz = dz;
    zGridInfo.Lz = Lz;
    

end