function plotInputTensors(xCellGrid,yCellGrid,zCellGrid,  txx_data,txy_data,txz_data,tyy_data,tyz_data,tzz_data,  flux_div_x_data,flux_div_y_data,flux_div_z_data,  hozAvg)

    % lets do a plot of the stress tensor txx, txy, txz, tyy, tyz, tzz and 
    % the flux_div_x, flux_div_y, and flux_div_z
    
    % correct the cell grids to be posOverL if they aren't already 
    %  posOverL grids
    [xOverL,x_nCells] = makePosOverL(xCellGrid);
    [yOverL,y_nCells] = makePosOverL(yCellGrid);
    [zOverL,z_nCells] = makePosOverL(zCellGrid);
    
    
    % setup plot data
    z_txx_plotVals = zeros(z_nCells,1);
    z_txy_plotVals = zeros(z_nCells,1);
    z_txz_plotVals = zeros(z_nCells,1);
    z_tyy_plotVals = zeros(z_nCells,1);
    z_tyz_plotVals = zeros(z_nCells,1);
    z_tzz_plotVals = zeros(z_nCells,1);
    z_flux_div_x_plotVals = zeros(z_nCells,1);
    z_flux_div_y_plotVals = zeros(z_nCells,1);
    z_flux_div_z_plotVals = zeros(z_nCells,1);
    txx_plotVals = zeros(z_nCells,1);
    txy_plotVals = zeros(z_nCells,1);
    txz_plotVals = zeros(z_nCells,1);
    tyy_plotVals = zeros(z_nCells,1);
    tyz_plotVals = zeros(z_nCells,1);
    tzz_plotVals = zeros(z_nCells,1);
    flux_div_x_plotVals = zeros(z_nCells,1);
    flux_div_y_plotVals = zeros(z_nCells,1);
    flux_div_z_plotVals = zeros(z_nCells,1);
    if hozAvg == true
        [z_txx_plotVals(:), txx_plotVals(:)] = hozAvg3D(txx_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_txy_plotVals(:), txy_plotVals(:)] = hozAvg3D(txy_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_txz_plotVals(:), txz_plotVals(:)] = hozAvg3D(txz_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_tyy_plotVals(:), tyy_plotVals(:)] = hozAvg3D(tyy_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_tyz_plotVals(:), tyz_plotVals(:)] = hozAvg3D(tyz_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_tzz_plotVals(:), tzz_plotVals(:)] = hozAvg3D(tzz_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_flux_div_x_plotVals(:), flux_div_x_plotVals(:)] = hozAvg3D(flux_div_x_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_flux_div_y_plotVals(:), flux_div_y_plotVals(:)] = hozAvg3D(flux_div_y_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_flux_div_z_plotVals(:), flux_div_z_plotVals(:)] = hozAvg3D(flux_div_z_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        lineStyle = 'kd-';
    else  % hozAvg == false
        z_txx_plotVals(:) = zOverL;
        z_txy_plotVals(:) = zOverL;
        z_txz_plotVals(:) = zOverL;
        z_tyy_plotVals(:) = zOverL;
        z_tyz_plotVals(:) = zOverL;
        z_tzz_plotVals(:) = zOverL;
        z_flux_div_x_plotVals(:) = zOverL;
        z_flux_div_y_plotVals(:) = zOverL;
        z_flux_div_z_plotVals(:) = zOverL;
        txx_plotVals(:) = txx_data;
        txy_plotVals(:) = txy_data;
        txz_plotVals(:) = txz_data;
        tyy_plotVals(:) = tyy_data;
        tyz_plotVals(:) = tyz_data;
        tzz_plotVals(:) = tzz_data;
        flux_div_x_plotVals(:) = flux_div_x_data;
        flux_div_y_plotVals(:) = flux_div_y_data;
        flux_div_z_plotVals(:) = flux_div_z_data;
        lineStyle = 'k.';
    end
    
    
    titleString = sprintf('input tensors');
    
    figure;
    subplot(3,3,1);
        plot(txx_plotVals,z_txx_plotVals,lineStyle);
        xlabel('txx');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,2);
        plot(txy_plotVals,z_txy_plotVals,lineStyle);
        xlabel('txy');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,3);
        plot(txz_plotVals,z_txz_plotVals,lineStyle);
        xlabel('txz');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,4);
        plot(tyy_plotVals,z_tyy_plotVals,lineStyle);
        xlabel('tyy');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,5);
        plot(tyz_plotVals,z_tyz_plotVals,lineStyle);
        xlabel('tyz');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,6);
        plot(tzz_plotVals,z_tzz_plotVals,lineStyle);
        xlabel('tzz');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,7);
        plot(flux_div_x_plotVals,z_flux_div_x_plotVals,lineStyle);
        xlabel('flux div x');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,8);
        plot(flux_div_y_plotVals,z_flux_div_y_plotVals,lineStyle);
        xlabel('flux div y');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,9);
        plot(flux_div_z_plotVals,z_flux_div_z_plotVals,lineStyle);
        xlabel('flux div z');
        ylabel("z\\L");
        ylim([0 1]);
            
    sgtitle(titleString);
    drawnow
    % adjust figure size
    fighandles = findall( allchild(0), 'type', 'figure');
    %%figPosition = get(fighandles(1),'position');
    set(fighandles(1),'Units', 'Normalized', 'OuterPosition', [0.1, 0.1, 0.8, 0.85]);
    pause(1);
    
    
end