function plotInputProfiles(uMeanLim,sigma2Lim,eppsLim,  xCellGrid,yCellGrid,zCellGrid,  uMean_data,vMean_data,wMean_data,sigma2_data,epps_data,  nonDim,hozAvg,  u_tao,del)

    % lets do a plot of the u, sigma2, and epps, 
    % then the v and w profiles
    % plotting nonDim profiles (nondimensional) can be a separate function
    % maybe, or a switch with the right if statement in here.
    % plotting stress tensor txx, txy, txz, tyy, tyz, tzz uses a separate
    % plot function
    % plotting initial calculated gradients and other fields will also be
    % done in a separate function
    
    
    % if uMeanLim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(uMeanLim)
        if size(uMeanLim,1) ~= 1 && size(uMeanLim,2) ~= 1
            error('!!! plotInputProfiles error !!! input uMeanLim is not a row or column vector !');
        end
        if length(uMeanLim) ~= 2
            error('!!! plotInputProfiles error !!! input uMeanLim is not size 2 !');
        end
        if uMeanLim(2) <= uMeanLim(1)
            error('!!! plotInputProfiles error !!! input uMeanLim 2nd posLim is not greater than 1st posLim !');
        end
    end
    
    % if sigma2Lim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(sigma2Lim)
        if size(sigma2Lim,1) ~= 1 && size(sigma2Lim,2) ~= 1
            error('!!! plotInputProfiles error !!! input sigma2Lim is not a row or column vector !');
        end
        if length(sigma2Lim) ~= 2
            error('!!! plotInputProfiles error !!! input sigma2Lim is not size 2 !');
        end
        if sigma2Lim(2) <= sigma2Lim(1)
            error('!!! plotInputProfiles error !!! input sigma2Lim 2nd posLim is not greater than 1st posLim !');
        end
    end
    
    % if eppsLim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(eppsLim)
        if size(eppsLim,1) ~= 1 && size(eppsLim,2) ~= 1
            error('!!! plotInputProfiles error !!! input eppsLim is not a row or column vector !');
        end
        if length(eppsLim) ~= 2
            error('!!! plotInputProfiles error !!! input eppsLim is not size 2 !');
        end
        if eppsLim(2) <= eppsLim(1)
            error('!!! plotInputProfiles error !!! input eppsLim 2nd posLim is not greater than 1st posLim !');
        end
    end
    
    
    % correct the cell grids to be posOverL if they aren't already 
    %  posOverL grids
    [xOverL,x_nCells] = makePosOverL(xCellGrid);
    [yOverL,y_nCells] = makePosOverL(yCellGrid);
    [zOverL,z_nCells] = makePosOverL(zCellGrid);
    
    
    % setup plot data
    z_uMean_plotVals = zeros(z_nCells,1);
    z_sigma2_plotVals = zeros(z_nCells,1);
    z_epps_plotVals = zeros(z_nCells,1);
    z_vMean_plotVals = zeros(z_nCells,1);
    z_wMean_plotVals = zeros(z_nCells,1);
    uMean_plotVals = zeros(z_nCells,1);
    sigma2_plotVals = zeros(z_nCells,1);
    epps_plotVals = zeros(z_nCells,1);
    vMean_plotVals = zeros(z_nCells,1);
    wMean_plotVals = zeros(z_nCells,1);
    if hozAvg == true
        [z_uMean_plotVals(:), uMean_plotVals(:)] = hozAvg3D(uMean_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_sigma2_plotVals(:), sigma2_plotVals(:)] = hozAvg3D(sigma2_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_epps_plotVals(:), epps_plotVals(:)] = hozAvg3D(epps_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_vMean_plotVals(:), vMean_plotVals(:)] = hozAvg3D(vMean_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        [z_wMean_plotVals(:), wMean_plotVals(:)] = hozAvg3D(wMean_data,"z",zCellGrid,x_nCells,y_nCells,z_nCells);
        lineStyle = 'kd-';
    else  % hozAvg == false
        z_uMean_plotVals(:) = zOverL;
        z_sigma2_plotVals(:) = zOverL;
        z_epps_plotVals(:) = zOverL;
        z_vMean_plotVals(:) = zOverL;
        z_wMean_plotVals(:) = zOverL;
        uMean_plotVals(:) = uMean_data;
        sigma2_plotVals(:) = sigma2_data;
        epps_plotVals(:) = epps_data;
        vMean_plotVals(:) = vMean_data;
        wMean_plotVals(:) = wMean_data;
        lineStyle = 'k.';
    end
    if nonDim == true
        uMean_plotVals = uMean_plotVals/u_tao;
        sigma2_plotVals = sigma2_plotVals/u_tao^2;
        epps_plotVals = epps_plotVals*del/u_tao^3;
    end
    
    
    titleString = sprintf('input velocity and other profiles');
    
    figure;
    subplot(2,3,1);
        plot(uMean_plotVals,z_uMean_plotVals,lineStyle);
        if nonDim == false
            xlabel('$\bar{u}$','Interpreter','Latex');
        else    % if nonDim == true
            xlabel('$\bar{u}$/$u_{\tau}$','Interpreter','Latex');
        end
        if ~isstring(uMeanLim)
            xlim(uMeanLim);
        end
        ylabel("z\\L");
        ylim([0 1]);
    subplot(2,3,2);
        plot(sigma2_plotVals,z_sigma2_plotVals,lineStyle);
        if nonDim == false
            xlabel('$\sigma^{2}$','Interpreter','Latex');
        else    % if nonDim == true
            xlabel('$\sigma^{2}$/$u^{2}_{\tau}$','Interpreter','Latex');
        end
        if ~isstring(sigma2Lim)
            xlim(sigma2Lim);
        end
        ylabel("z\\L");
        ylim([0 1]);
    subplot(2,3,3);
        plot(epps_plotVals,z_epps_plotVals,lineStyle);
        if nonDim == false
            xlabel('$\bar{\epsilon}$','Interpreter','Latex');
        else    % if nonDim == true
            xlabel('$\bar{\epsilon}\delta$/$u^{3}_{\tau}$','Interpreter','Latex');
        end
        if ~isstring(eppsLim)
            xlim(eppsLim);
        end
        ylabel("z\\L");
        ylim([0 1]);
        
    subplot(2,3,4);
        plot(vMean_plotVals,z_vMean_plotVals,lineStyle);
        xlabel('$\bar{v}$','Interpreter','Latex');
        ylabel("z\\L");
        ylim([0 1]);
    subplot(2,3,5);
        plot(wMean_plotVals,z_wMean_plotVals,lineStyle);
        xlabel('$\bar{w}$','Interpreter','Latex');
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