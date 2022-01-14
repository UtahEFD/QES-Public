function plotStatisticsFigure_LES(figTitle,dimName,nStatisticBins,  zLim,subfilter_tke_Lim,uFluct_wFluct_covariances_Lim,delta_wFluct_variances_Lim,  fileExists_array,  plotBasename_array,  C_0,tzz_data,txz_data,epps_data,xCellGrid,yCellGrid,zCellGrid,dataExists,  current_time_array,timestep_array,  current_isActive_array,  current_pos_array,uFluct_array,wFluct_array,delta_wFluct_array)

    if ~(dimName == "x" || dimName == "y" || dimName == "z")
        error('!!! plotStatisticsFigure_LES error !!! input dimName is not \"x\", \"y\", or \"z\" but is \"%s\" !!!',dimName);
    end
    
    if size(fileExists_array,1) ~= 1 && size(fileExists_array,2) ~= 1
        error('!!! plotStatisticsFigure_LES error !!! input fileExists_array is not a row or a column vector!');
    end
    nDatasets = length(fileExists_array);

    
    % if zLim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(zLim)
        if size(zLim,1) ~= 1 && size(zLim,2) ~= 1
            error('!!! plotStatisticsFigure_LES error !!! input zLim is not a row or column vector !');
        end
        if length(zLim) ~= 2
            error('!!! plotStatisticsFigure_LES error !!! input zLim is not size 2 !');
        end
        if zLim(2) <= zLim(1)
            error('!!! plotStatisticsFigure_LES error !!! input zLim 2nd Lim is not greater than 1st Lim !');
        end
    end
    
    
    % if subfilter_tke_Lim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(subfilter_tke_Lim)
        if size(subfilter_tke_Lim,1) ~= 1 && size(subfilter_tke_Lim,2) ~= 1
            error('!!! plotStatisticsFigure_LES error !!! input subfilter_tke_Lim is not a row or column vector !');
        end
        if length(subfilter_tke_Lim) ~= 2
            error('!!! plotStatisticsFigure_LES error !!! input subfilter_tke_Lim is not size 2 !');
        end
        if subfilter_tke_Lim(2) <= subfilter_tke_Lim(1)
            error('!!! plotStatisticsFigure_LES error !!! input subfilter_tke_Lim 2nd Lim is not greater than 1st Lim !');
        end
    end
    
    % if uFluct_wFluct_covariances_Lim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(uFluct_wFluct_covariances_Lim)
        if size(uFluct_wFluct_covariances_Lim,1) ~= 1 && size(uFluct_wFluct_covariances_Lim,2) ~= 1
            error('!!! plotStatisticsFigure_LES error !!! input uFluct_wFluct_covariances_Lim is not a row or column vector !');
        end
        if length(uFluct_wFluct_covariances_Lim) ~= 2
            error('!!! plotStatisticsFigure_LES error !!! input uFluct_wFluct_covariances_Lim is not size 2 !');
        end
        if uFluct_wFluct_covariances_Lim(2) <= uFluct_wFluct_covariances_Lim(1)
            error('!!! plotStatisticsFigure_LES error !!! input uFluct_wFluct_covariances_Lim 2nd Lim is not greater than 1st Lim !');
        end
    end
    
    % if delta_wFluct_variances_Lim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(delta_wFluct_variances_Lim)
        if size(delta_wFluct_variances_Lim,1) ~= 1 && size(delta_wFluct_variances_Lim,2) ~= 1
            error('!!! plotStatisticsFigure_LES error !!! input delta_wFluct_variances_Lim is not a row or column vector !');
        end
        if length(delta_wFluct_variances_Lim) ~= 2
            error('!!! plotStatisticsFigure_LES error !!! input delta_wFluct_variances_Lim is not size 2 !');
        end
        if delta_wFluct_variances_Lim(2) <= delta_wFluct_variances_Lim(1)
            error('!!! plotStatisticsFigure_LES error !!! input delta_wFluct_variances_Lim 2nd Lim is not greater than 1st Lim !');
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
        error('!!! plotStatisticsFigure_LES error !!! input dataExists for the input expected value data is false!');
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
        expectedVals_txz = zeros(expectedVals_z_nCells,1);
        expectedVals_epps = zeros(expectedVals_z_nCells,1);
        if hozAvg == true
            [ ~, expectedVals_tzz ] = hozAvg3D(tzz_data,"z",zCellGrid,expectedVals_x_nCells,expectedVals_y_nCells,expectedVals_z_nCells);
            [ ~, expectedVals_txz ] = hozAvg3D(txz_data,"z",zCellGrid,expectedVals_x_nCells,expectedVals_y_nCells,expectedVals_z_nCells);
            [ ~, expectedVals_epps ] = hozAvg3D(epps_data,"z",zCellGrid,expectedVals_x_nCells,expectedVals_y_nCells,expectedVals_z_nCells);
        else
            expectedVals_tzz(:) = tzz_data;
            expectedVals_txz(:) = txz_data;
            expectedVals_epps(:) = epps_data;
        end
        
        expected_subfilter_tke = expectedVals_tzz;   % expect it to be the variance
        expected_uFluct_wFluct_covariances = expectedVals_txz;   % expect it to be the covariance
        expected_delta_wFluct_variances = C_0*expectedVals_epps;   % expect it to be C0*epps*timestep, though for some odd reason it doesn't seem to need the timestep
        
    end
    
    % now calculate found particle statistics
    subfilter_tke = zeros(nDatasets,nStatisticBins);
    uFluct_wFluct_covariances = zeros(nDatasets,nStatisticBins);
    delta_wFluct_variances = zeros(nDatasets,nStatisticBins);
    for idx = 1:nDatasets
        
        if fileExists_array(idx) == true
            % notice that it is current_ as opposed to old_ values
            current_isActive_indices = find(cell2mat(current_isActive_array(idx)) == true);
            if ~(isempty(current_isActive_indices))

                % grab the required values for plotting
                % first pull out cell array stuff
                timestep = cell2mat(timestep_array(idx));
                
                current_pos = cell2mat(current_pos_array(idx));
                current_uFluct = cell2mat(uFluct_array(idx));
                current_wFluct = cell2mat(wFluct_array(idx));
                current_delta_wFluct = cell2mat(delta_wFluct_array(idx));
                
                % now pull out the required indices of values
                current_pos_current_isActive = current_pos(current_isActive_indices);
                current_uFluct_current_isActive = current_uFluct(current_isActive_indices);
                current_wFluct_current_isActive = current_wFluct(current_isActive_indices);
                current_delta_wFluct_current_isActive = current_delta_wFluct(current_isActive_indices);
                
                % correct the grid to be posOverL if isn't already posOverL
                [current_posOverL_current_isActive,z_nCells] = makePosOverL(current_pos_current_isActive);

                %%% get the indices of a given variable
                [current_N,current_edges,current_bin_indices] = histcounts(current_posOverL_current_isActive,statisticsGridEdges_posOverL);

                %%% now calculate the required statistics
                for binIdx = 1:nStatisticBins
                    subfilter_tke(idx,binIdx) = var(current_uFluct_current_isActive(current_bin_indices == binIdx));
                    currentCovariances = cov(current_uFluct_current_isActive(current_bin_indices == binIdx),current_wFluct_current_isActive(current_bin_indices == binIdx));
                    uFluct_wFluct_covariances(idx,binIdx) = currentCovariances(1,2);    % 1,2 comes from matlab help describing the output as covar(A,A), covar(A,B), covar(B,A), covar(B,B). This should be covar(A,B).
                    delta_wFluct_variances(idx,binIdx) = var(current_delta_wFluct_current_isActive(current_bin_indices == binIdx))/timestep;
                end

            end     % if is active indices
            
        end     % if file exists
    end     % nPlots for loop
    
    colormarkers = [
        ":.",
        "^",
        "s",
        "o",
        "h"
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
        24,
        4.5,
        6,
        4.5,
        4.5
        ];
    figure;
    subplot(2,3,1);
        plot(expected_subfilter_tke,expectedVals_zOverL,colormarkers(1),'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),'LineWidth',linewidth(1),'MarkerSize',markersize(1));
        hold on;
        for idx = 1:nDatasets
            if fileExists_array(idx) == true
                plot(subfilter_tke(idx,:),statisticsGrid_posOverL,colormarkers(idx+1),'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
            end
        end
        xlabel(sprintf("subfilter tke"));
        if ~isstring(subfilter_tke_Lim)
            xlim(subfilter_tke_Lim);
        end
        ylabel(sprintf("%s\\\\L",dimName));
        if ~isstring(zLim)
            ylim(zLim);
        end
        hold off;
    subplot(2,3,2);
        plot(expected_uFluct_wFluct_covariances,expectedVals_zOverL,colormarkers(1),'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),'LineWidth',linewidth(1),'MarkerSize',markersize(1));
        hold on;
        for idx = 1:nDatasets
            if fileExists_array(idx) == true
                plot(uFluct_wFluct_covariances(idx,:),statisticsGrid_posOverL,colormarkers(idx+1),'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
            end
        end
        xlabel(sprintf("uFluct wFluct covariances"));
        if ~isstring(uFluct_wFluct_covariances_Lim)
            xlim(uFluct_wFluct_covariances_Lim);
        end
        ylabel(sprintf("%s\\\\L",dimName));
        if ~isstring(zLim)
            ylim(zLim);
        end
        hold off;
    subplot(2,3,3);
        plot(expected_delta_wFluct_variances,expectedVals_zOverL,colormarkers(1),'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),'LineWidth',linewidth(1),'MarkerSize',markersize(1));
        hold on;
        for idx = 1:nDatasets
            if fileExists_array(idx) == true
                plot(delta_wFluct_variances(idx,:),statisticsGrid_posOverL,colormarkers(idx+1),'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
            end
        end
        xlabel(sprintf("wFluct increment variances\ndivided by timestep"));
        if ~isstring(delta_wFluct_variances_Lim)
            xlim(delta_wFluct_variances_Lim);
        end
        ylabel(sprintf("%s\\\\L",dimName));
        if ~isstring(zLim)
            ylim(zLim);
        end
        hold off;
    subplot(2,3,4:6);   % this will be for the legend
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