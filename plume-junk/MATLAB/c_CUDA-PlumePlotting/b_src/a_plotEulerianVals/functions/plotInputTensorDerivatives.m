function plotInputTensorDerivatives(xCellGrid,yCellGrid,zCellGrid,  dtxxdx_data,dtxydx_data,dtxzdx_data,dtxydy_data,dtyydy_data,dtyzdy_data,dtxzdz_data,dtyzdz_data,dtzzdz_data,  hozAvg)

    % lets do a plot of the tensor derivatives used to calculate the flux
    % div variables
    
    % correct the cell grids to be posOverL if they aren't already 
    %  posOverL grids
    [xOverL,x_nCells] = makePosOverL(xCellGrid);
    [yOverL,y_nCells] = makePosOverL(yCellGrid);
    [zOverL,z_nCells] = makePosOverL(zCellGrid);
    
    
    % setup plot data
    z_dtxxdx_plotVals = zeros(z_nCells,1);
    z_dtxydx_plotVals = zeros(z_nCells,1);
    z_dtxzdx_plotVals = zeros(z_nCells,1);
    z_dtxydy_plotVals = zeros(z_nCells,1);
    z_dtyydy_plotVals = zeros(z_nCells,1);
    z_dtyzdy_plotVals = zeros(z_nCells,1);
    z_dtxzdz_plotVals = zeros(z_nCells,1);
    z_dtyzdz_plotVals = zeros(z_nCells,1);
    z_dtzzdz_plotVals = zeros(z_nCells,1);
    dtxxdx_plotVals = zeros(z_nCells,1);
    dtxydx_plotVals = zeros(z_nCells,1);
    dtxzdx_plotVals = zeros(z_nCells,1);
    dtxydy_plotVals = zeros(z_nCells,1);
    dtyydy_plotVals = zeros(z_nCells,1);
    dtyzdy_plotVals = zeros(z_nCells,1);
    dtxzdz_plotVals = zeros(z_nCells,1);
    dtyzdz_plotVals = zeros(z_nCells,1);
    dtzzdz_plotVals = zeros(z_nCells,1);
    if hozAvg == true
        [z_dtxxdx_plotVals(:), dtxxdx_plotVals(:)] = hozAvg3D(dtxxdx_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_dtxydx_plotVals(:), dtxydx_plotVals(:)] = hozAvg3D(dtxydx_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_dtxzdx_plotVals(:), dtxzdx_plotVals(:)] = hozAvg3D(dtxzdx_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_dtxydy_plotVals(:), dtxydy_plotVals(:)] = hozAvg3D(dtxydy_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_dtyydy_plotVals(:), dtyydy_plotVals(:)] = hozAvg3D(dtyydy_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_dtyzdy_plotVals(:), dtyzdy_plotVals(:)] = hozAvg3D(dtyzdy_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_dtxzdz_plotVals(:), dtxzdz_plotVals(:)] = hozAvg3D(dtxzdz_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_dtyzdz_plotVals(:), dtyzdz_plotVals(:)] = hozAvg3D(dtyzdz_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_dtzzdz_plotVals(:), dtzzdz_plotVals(:)] = hozAvg3D(dtzzdz_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        lineStyle = 'kd-';
    else  % hozAvg == false
        z_dtxxdx_plotVals(:) = zOverL;
        z_dtxydx_plotVals(:) = zOverL;
        z_dtxzdx_plotVals(:) = zOverL;
        z_dtxydy_plotVals(:) = zOverL;
        z_dtyydy_plotVals(:) = zOverL;
        z_dtyzdy_plotVals(:) = zOverL;
        z_dtxzdz_plotVals(:) = zOverL;
        z_dtyzdz_plotVals(:) = zOverL;
        z_dtzzdz_plotVals(:) = zOverL;
        dtxxdx_plotVals(:) = dtxxdx_data;
        dtxydx_plotVals(:) = dtxydx_data;
        dtxzdx_plotVals(:) = dtxzdx_data;
        dtxydy_plotVals(:) = dtxydy_data;
        dtyydy_plotVals(:) = dtyydy_data;
        dtyzdy_plotVals(:) = dtyzdy_data;
        dtxzdz_plotVals(:) = dtxzdz_data;
        dtyzdz_plotVals(:) = dtyzdz_data;
        dtzzdz_plotVals(:) = dtzzdz_data;
        lineStyle = 'k.';
    end
    
    
    titleString = sprintf('input tensor derivatives');
    
    figure;
    subplot(3,3,1);
        plot(dtxxdx_plotVals,z_dtxxdx_plotVals,lineStyle);
        xlabel('dtxxdx');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,2);
        plot(dtxydx_plotVals,z_dtxydx_plotVals,lineStyle);
        xlabel('dtxydx');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,3);
        plot(dtxzdx_plotVals,z_dtxzdx_plotVals,lineStyle);
        xlabel('dtxzdx');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,4);
        plot(dtxydy_plotVals,z_dtxydy_plotVals,lineStyle);
        xlabel('dtxydy');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,5);
        plot(dtyydy_plotVals,z_dtyydy_plotVals,lineStyle);
        xlabel('dtyydy');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,6);
        plot(dtyzdy_plotVals,z_dtyzdy_plotVals,lineStyle);
        xlabel('dtyzdy');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,7);
        plot(dtxzdz_plotVals,z_dtxzdz_plotVals,lineStyle);
        xlabel('dtxzdz');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,8);
        plot(dtyzdz_plotVals,z_dtyzdz_plotVals,lineStyle);
        xlabel('dtyzdz');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(3,3,9);
        plot(dtzzdz_plotVals,z_dtzzdz_plotVals,lineStyle);
        xlabel('dtzzdz');
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