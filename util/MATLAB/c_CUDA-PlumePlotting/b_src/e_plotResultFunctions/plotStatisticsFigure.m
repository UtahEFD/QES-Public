function plotStatisticsFigure(figTitle,dimName,nStatisticBins,  velFluct_averages_Lim,velFluct_variances_Lim,delta_velFluct_averages_Lim,delta_velFluct_variances_Lim,  fileExists_array,  plotBasename_array,  C_0,tzz_data,dtzzdz_data,epps_data,xCellGrid,yCellGrid,zCellGrid,dataExists,  current_time_array,timestep_array,  current_isActive_array,  current_pos_array,velFluct_array,delta_velFluct_array)

    if ~(dimName == "x" || dimName == "y" || dimName == "z")
        error('!!! plotStatisticsFigure error !!! input dimName is not \"x\", \"y\", or \"z\" but is \"%s\" !!!',dimName);
    end

    if size(fileExists_array,1) ~= 1 && size(fileExists_array,2) ~= 1
        error('!!! plotStatisticsFigure error !!! input fileExists_array is not a row or a column vector!');
    end
    nDatasets = length(fileExists_array);

    
    % if velFluct_averages_Lim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(velFluct_averages_Lim)
        if size(velFluct_averages_Lim,1) ~= 1 && size(velFluct_averages_Lim,2) ~= 1
            error('!!! plotStatisticsFigure error !!! input velFluct_averages_Lim is not a row or column vector !');
        end
        if length(velFluct_averages_Lim) ~= 2
            error('!!! plotStatisticsFigure error !!! input velFluct_averages_Lim is not size 2 !');
        end
        if velFluct_averages_Lim(2) <= velFluct_averages_Lim(1)
            error('!!! plotStatisticsFigure error !!! input velFluct_averages_Lim 2nd Lim is not greater than 1st Lim !');
        end
    end
    
    % if velFluct_variances_Lim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(velFluct_variances_Lim)
        if size(velFluct_variances_Lim,1) ~= 1 && size(velFluct_variances_Lim,2) ~= 1
            error('!!! plotStatisticsFigure error !!! input velFluct_variances_Lim is not a row or column vector !');
        end
        if length(velFluct_variances_Lim) ~= 2
            error('!!! plotStatisticsFigure error !!! input velFluct_variances_Lim is not size 2 !');
        end
        if velFluct_variances_Lim(2) <= velFluct_variances_Lim(1)
            error('!!! plotStatisticsFigure error !!! input velFluct_variances_Lim 2nd Lim is not greater than 1st Lim !');
        end
    end
    
    % if delta_velFluct_averages_Lim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(delta_velFluct_averages_Lim)
        if size(delta_velFluct_averages_Lim,1) ~= 1 && size(delta_velFluct_averages_Lim,2) ~= 1
            error('!!! plotStatisticsFigure error !!! input delta_velFluct_averages_Lim is not a row or column vector !');
        end
        if length(delta_velFluct_averages_Lim) ~= 2
            error('!!! plotStatisticsFigure error !!! input delta_velFluct_averages_Lim is not size 2 !');
        end
        if delta_velFluct_averages_Lim(2) <= delta_velFluct_averages_Lim(1)
            error('!!! plotStatisticsFigure error !!! input delta_velFluct_averages_Lim 2nd Lim is not greater than 1st Lim !');
        end
    end
    
    % if delta_velFluct_variances_Lim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(delta_velFluct_variances_Lim)
        if size(delta_velFluct_variances_Lim,1) ~= 1 && size(delta_velFluct_variances_Lim,2) ~= 1
            error('!!! plotStatisticsFigure error !!! input delta_velFluct_variances_Lim is not a row or column vector !');
        end
        if length(delta_velFluct_variances_Lim) ~= 2
            error('!!! plotStatisticsFigure error !!! input delta_velFluct_variances_Lim is not size 2 !');
        end
        if delta_velFluct_variances_Lim(2) <= delta_velFluct_variances_Lim(1)
            error('!!! plotStatisticsFigure error !!! input delta_velFluct_variances_Lim 2nd Lim is not greater than 1st Lim !');
        end
    end
    
    
    % setup the viewingGrid
    %%% maybe setup zeros for the other two dimensions instead of an equal
    %%% linspace? I guess technically this is separate from the actual
    %%% grid, we just need a number of points in each direction. So using
    %%% this grid for all three dimensions probably should work better than
    %%% zeros for the other two dimensions.
    nStatisticEdges = nStatisticBins+1;   % edges are always one index more, cause it is the faces of the bins
    statisticsGrid_posOverL = linspace(0,1,nStatisticBins);
    statisticsGridEdges_posOverL = linspace(0,1,nStatisticEdges);
    
    
    % now calculate the estimated turbulence statistics for later comparison
    % going to assume the C_0, and all function handles in the input array
    % are equal to the first value
    if dataExists == false
        error('!!! plotStatisticsFigure error !!! input dataExists for the input expected value data is false!');
    else
            
        % correct the cell grids to be posOverL if they aren't already 
        %  posOverL grids
        [expectedVals_xOverL,expectedVals_x_nCells] = makePosOverL(xCellGrid);
        [expectedVals_yOverL,expectedVals_y_nCells] = makePosOverL(yCellGrid);
        [expectedVals_zOverL,expectedVals_z_nCells] = makePosOverL(zCellGrid);

        % set hozAvg to be true IF the input data is 3D
        if (expectedVals_x_nCells ~= 1 && expectedVals_y_nCells ~= 1) || (expectedVals_x_nCells ~= 1 && expectedVals_z_nCells ~= 1) || (expectedVals_y_nCells ~= 1 && expectedVals_z_nCells ~= 1)
            hozAvg = true;
        else
            hozAvg = false;
        end
        
        expectedVals_tzz = zeros(expectedVals_z_nCells,1);
        expectedVals_dtzzdz = zeros(expectedVals_z_nCells,1);
        expectedVals_epps = zeros(expectedVals_z_nCells,1);
        if hozAvg == true
            [ ~, expectedVals_tzz(:) ] = hozAvg3D(tzz_data,"z",zCellGrid,expectedVals_x_nCells,expectedVals_y_nCells,expectedVals_z_nCells);
            [ ~, expectedVals_dtzzdz(:) ] = hozAvg3D(dtzzdz_data,"z",zCellGrid,expectedVals_x_nCells,expectedVals_y_nCells,expectedVals_z_nCells);
            [ ~, expectedVals_epps(:) ] = hozAvg3D(epps_data,"z",zCellGrid,expectedVals_x_nCells,expectedVals_y_nCells,expectedVals_z_nCells);
        else
            expectedVals_tzz(:) = tzz_data;
            expectedVals_dtzzdz(:) = dtzzdz_data;
            expectedVals_epps(:) = epps_data;
        end
        
        expected_velFluct_averages = zeros(expectedVals_z_nCells,1);  % expect it to be zero or the mean of the wind, not sure which yet
        expected_velFluct_variances = expectedVals_tzz;   % expect it to be the variance
        expected_delta_velFluct_averages = expectedVals_dtzzdz;  % expect it to be d(sigma^2)/dx
        expected_delta_velFluct_variances = C_0*expectedVals_epps;   % expect it to be C0*epps*timestep, though for some odd reason it doesn't seem to need the timestep
        
    end
    
    % now calculate found particle statistics
    velFluct_averages = zeros(nDatasets,nStatisticBins);
    velFluct_variances = zeros(nDatasets,nStatisticBins);
    delta_velFluct_averages = zeros(nDatasets,nStatisticBins);
    delta_velFluct_variances = zeros(nDatasets,nStatisticBins);
    for idx = 1:nDatasets
        
        if fileExists_array(idx) == true
            % notice that it is current_ as opposed to old_ values
            current_isActive_indices = find(cell2mat(current_isActive_array(idx)) == true);
            if ~(isempty(current_isActive_indices))

                % grab the required values for plotting
                % first pull out cell array stuff
                timestep = cell2mat(timestep_array(idx));
                
                current_pos = cell2mat(current_pos_array(idx));
                current_velFluct = cell2mat(velFluct_array(idx));
                current_delta_velFluct = cell2mat(delta_velFluct_array(idx));
                
                % now pull out the required indices of values
                current_pos_current_isActive = current_pos(current_isActive_indices);
                current_velFluct_current_isActive = current_velFluct(current_isActive_indices);
                current_delta_velFluct_current_isActive = current_delta_velFluct(current_isActive_indices);
                
                % correct the grid to be posOverL if isn't already posOverL
                [current_posOverL_current_isActive,z_nCells] = makePosOverL(current_pos_current_isActive);

                %%% get the indices of a given variable
                [current_N,current_edges,current_bin_indices] = histcounts(current_posOverL_current_isActive,statisticsGridEdges_posOverL);

                %%% now calculate the required statistics
                for binIdx = 1:nStatisticBins
                    velFluct_averages(idx,binIdx) = mean(current_velFluct_current_isActive(current_bin_indices == binIdx));
                    velFluct_variances(idx,binIdx) = var(current_velFluct_current_isActive(current_bin_indices == binIdx));
                    %delta_velFluct_averages(idx,binIdx) = mean(currentDeltaXs(current_bin_indices == binIdx))/(timestep^2);
                    delta_velFluct_averages(idx,binIdx) = mean(current_delta_velFluct_current_isActive(current_bin_indices == binIdx))/timestep;
                    %delta_velFluct_variances(idx,binIdx) = var(currentDeltaXs(current_bin_indices == binIdx))/(timestep^2);
                    delta_velFluct_variances(idx,binIdx) = var(current_delta_velFluct_current_isActive(current_bin_indices == binIdx))/timestep;
                end

            end     % if is active indices
            
        end     % if file exists
    end     % nPlots for loop
    
    colormarkers = [
        ".",
        "--",
        "x-",
        "+-",
        "--"
        ];
    colormap = [
        [1, 0, 0];
        [0, 0, 1];
        [0, 0.5, 0];
        [0.6350, 0.0780, 0.1840];
        [0, 0, 0];
        ];
    linewidth = [
        1.5,
        1.5,
        1.5,
        1.5,
        1.5
        ];
    markersize = [
        12,
        6,
        6,
        6,
        6
        ];
    figure;
    subplot(3,2,1);
        plot(expected_velFluct_averages,expectedVals_zOverL,colormarkers(1),'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),'LineWidth',linewidth(1),'MarkerSize',markersize(1));
        hold on;
        for idx = 1:nDatasets
            if fileExists_array(idx) == true
                plot(velFluct_averages(idx,:),statisticsGrid_posOverL,colormarkers(idx+1),'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
            end
        end
        xlabel(sprintf("average particle velocity\n(mean of binned velocity fluctuations)"));
        if ~isstring(velFluct_averages_Lim)
            xlim(velFluct_averages_Lim);
        end
        ylabel(sprintf("%s\\\\L",dimName));
        ylim([0 1]);
        hold off;
    subplot(3,2,2);
        plot(expected_velFluct_variances,expectedVals_zOverL,colormarkers(1),'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),'LineWidth',linewidth(1),'MarkerSize',markersize(1));
        hold on;
        for idx = 1:nDatasets
            if fileExists_array(idx) == true
                plot(velFluct_variances(idx,:),statisticsGrid_posOverL,colormarkers(idx+1),'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
            end
        end
        xlabel(sprintf("particle velocity variance\n(variance of binned velocity fluctuations)"));
        if ~isstring(velFluct_variances_Lim)
            xlim(velFluct_variances_Lim);
        end
        ylabel(sprintf("%s\\\\L",dimName));
        ylim([0 1]);
        hold off;
    subplot(3,2,3);
        plot(expected_delta_velFluct_averages,expectedVals_zOverL,colormarkers(1),'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),'LineWidth',linewidth(1),'MarkerSize',markersize(1));
        hold on;
        for idx = 1:nDatasets
            if fileExists_array(idx) == true
                plot(delta_velFluct_averages(idx,:),statisticsGrid_posOverL,colormarkers(idx+1),'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
            end
        end
        xlabel(sprintf("average particle acceleration\n(mean of binned delta velocity fluctuations divided by time)"));
        if ~isstring(delta_velFluct_averages_Lim)
            xlim(delta_velFluct_averages_Lim);
        end
        ylabel(sprintf("%s\\\\L",dimName));
        ylim([0 1]);
        hold off;
    subplot(3,2,4);
        plot(expected_delta_velFluct_variances,expectedVals_zOverL,colormarkers(1),'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),'LineWidth',linewidth(1),'MarkerSize',markersize(1));
        hold on;
        for idx = 1:nDatasets
            if fileExists_array(idx) == true
                plot(delta_velFluct_variances(idx,:),statisticsGrid_posOverL,colormarkers(idx+1),'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
            end
        end
        xlabel(sprintf("variance of particle velocity increments\n(variance of binned delta velocity fluctuations divided by time)"));
        if ~isstring(delta_velFluct_variances_Lim)
            xlim(delta_velFluct_variances_Lim);
        end
        ylabel(sprintf("%s\\\\L",dimName));
        ylim([0 1]);
        hold off;
    subplot(3,2,5:6);   % this will be for the legend
        plot(nan,nan,colormarkers(1),'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),'LineWidth',linewidth(1),'MarkerSize',markersize(1));
        hold on;
        for idx = 1:nDatasets
            if fileExists_array(idx) == true
                plot(nan,nan,colormarkers(idx+1),'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
            end
        end
        axis off;
        legendString = strings(nDatasets+1,1);
        legendString(1) = 'exact solution';
        for idx = 1:nDatasets
            if fileExists_array(idx) == true
                % first pull out the values from the cell array
                current_time = cell2mat(current_time_array(idx));
                timestep = cell2mat(timestep_array(idx));
                plotBasename = string(plotBasename_array(idx));
                legendString(idx+1) = sprintf('time = %0.4f, dt = %0.4f, %s',current_time,timestep,plotBasename);
            end
        end
        ledgerend = legend(legendString,'Location','North');
        set(ledgerend,'FontSize',12);
        hold off;
        
    sgtitle(figTitle);
    drawnow
    % adjust figure size
    fighandles = findall( allchild(0), 'type', 'figure');
    %%figPosition = get(fighandles(1),'position');
    set(fighandles(1),'Units', 'Normalized', 'OuterPosition', [0.1, 0.1, 0.8, 0.85]);
    
end