function plotStatisticsFigure( caseBaseName,plotOutputDir,show_ghostCells, ...
    wFluct_averages_Lim,wFluct_variances_Lim,delta_wFluct_averages_Lim,delta_wFluct_variances_Lim,...
    nBins,currentTime_array,timestep_array,zGridInfo,turb_data,caseNames,plumeParInfo_data,C0)

%%% before anything else, pull out the data needed from the packed dataset structures

% pull out the eulerian grid values
zOverL_cc = zGridInfo.zOverL_cc;
nz_cc = zGridInfo.nz_cc;
dz = zGridInfo.dz;
Lz = zGridInfo.Lz;


% now pull out the turb data
tzz = turb_data.tzz;

CoEps = turb_data.CoEps;
epps = CoEps/C0;


% get the number of datasets to plot
nDatasets = length(caseNames);

% pull out some of the data from the plume datasets
for fileIdx = 1:nDatasets
    
    % grab all par data from the very last timeset, nothing else
    % data is in format (nPar,nTimes) for each dataset in a case
    isActive_array.(caseNames(fileIdx)) = plumeParInfo_data.(caseNames(fileIdx)).isActive(:,end);
    
    zPos_array.(caseNames(fileIdx)) = plumeParInfo_data.(caseNames(fileIdx)).zPos(:,end);
    
    wFluct_array.(caseNames(fileIdx)) = plumeParInfo_data.(caseNames(fileIdx)).wFluct(:,end);
    delta_wFluct_array.(caseNames(fileIdx)) = plumeParInfo_data.(caseNames(fileIdx)).delta_wFluct(:,end);
    
end


%%% first, calculate the expected values
%%% this includes taking the eulerian data and horizontally averaging it
%%% this also includes calculating a derivative of the variance

% calculate the derivative of the variance
dtzzdz = calcDerivative(tzz,dz);

% now calculate the horizontally averaged eularian values
[ tzz_hozAvged ] = hozAvg3D( tzz );
[ epps_hozAvged ] = hozAvg3D( epps );
[ dtzzdz_hozAvged ] = hozAvg3D( dtzzdz );

% now use all the above to set the expected particle statistic values
expected_wFluct_averages = zeros(nz_cc,1);  % expect it to be zero or the mean of the wind, not sure which yet
expected_wFluct_variances = tzz_hozAvged;   % expect it to be the variance, which is the same as the diagonal of the stress tensor
expected_delta_wFluct_averages = dtzzdz_hozAvged;  % expect it to be d(sigma^2)/dz, after "dividing" by timestep on paper it's this
expected_delta_wFluct_variances = C0*epps_hozAvged;   % expect it to be C0*epps*timestep, after "dividing" by timestep on paper it's this



%%% now calculate the found particle statistics

% create the edge grid, to make histcounts easier and more correct
nEdges = nBins+1;   % edges are always one index more, cause it is the faces of the bins
gridEdges_zOverL = linspace(0,1,nEdges);    % for statistic bins

% make the statistics plotting grid, for plotting statistics. Note that
% the histcounts uses grid edges, but outputs at bin centers.
statistics_dz = gridEdges_zOverL(2) - gridEdges_zOverL(1);
% notice that linspace(statistics_dz/2,1-statistics_dz/2,nBins) should
% get the same result as this.
statisticsGrid_zOverL = statistics_dz/2:statistics_dz:1-statistics_dz/2;


% set the found particle statistics dataset sizes
wFluct_averages = nan(nDatasets,nBins);
wFluct_variances = nan(nDatasets,nBins);
delta_wFluct_averages = nan(nDatasets,nBins);
delta_wFluct_variances = nan(nDatasets,nBins);

% now calculate the particle statistics
for idx = 1:nDatasets
    
    % first get the current dataset data
    current_timestep = timestep_array(idx);
    
    current_isActive = isActive_array.(caseNames(idx));
    
    current_zPos = zPos_array.(caseNames(idx));
    current_wFluct = wFluct_array.(caseNames(idx));
    current_delta_wFluct = delta_wFluct_array.(caseNames(idx));
    
    
    % now get the indices for which particles are active
    current_isActive_indices = find( current_isActive == true );
    
    
    % now filter the current dataset data by active particles
    % (assumes rogue particles are already set to inactive)
    currentActive_zPos = current_zPos(current_isActive_indices);
    currentActive_wFluct = current_wFluct(current_isActive_indices);
    currentActive_delta_wFluct = current_delta_wFluct(current_isActive_indices);
    
    % correct the particle position values to be zOverL
    currentActive_zPosOverL = currentActive_zPos/Lz;
    
    % get the indices of a given variable
    [current_N,current_edges,current_bin_indices] = histcounts(currentActive_zPosOverL,gridEdges_zOverL);
    
    
    %%% now calculate the required statistics
    for binIdx = 1:nBins
        wFluct_averages(idx,binIdx) = mean(currentActive_wFluct(current_bin_indices == binIdx));
        wFluct_variances(idx,binIdx) = var(currentActive_wFluct(current_bin_indices == binIdx));
        %delta_velFluct_averages(idx,binIdx) = mean(currentActive_delta_wFluct(current_bin_indices == binIdx))/(current_timestep^2);
        delta_wFluct_averages(idx,binIdx) = mean(currentActive_delta_wFluct(current_bin_indices == binIdx))/current_timestep;   % is divided by timestep on paper to get this eqn
        %delta_velFluct_variances(idx,binIdx) = var(currentActive_delta_wFluct(current_bin_indices == binIdx))/(current_timestep^2);
        delta_wFluct_variances(idx,binIdx) = var(currentActive_delta_wFluct(current_bin_indices == binIdx))/current_timestep;   % is divided by timestep on paper to get this eqn
    end
    
    
end



%%% now plot the expected values and the found particle statistics

% set the axis limits
% if ghost values are not to be shown, use 0 to 1.
% if ghost values are to be shown, use the largest possible axis limits
if show_ghostCells == false
    z_plotLim = [0 1];
else
    z_plotLim = [zOverL_cc(1) zOverL_cc(end)];
end

% set the title string
titleString = sprintf('Resulting Statistic Profiles');

% set some useful plot variables, for controlling the plot a bit
colormarkers = [
    ".",
    "-",
    "-",
    "-",
    "-"
    ];
colormap = [
    [1, 0, 0];
    [0, 0, 1];
    [0, 0.5, 0];
    [0.6350, 0.0780, 0.1840];
    [0, 0, 0];
    ];
linewidth = [
    1,
    1,
    1,
    1,
    1
    ];
markersize = [
    12,
    4,
    4,
    4,
    4
    ];

% set the subplot size
nSubplotRows = 1;
nSubplotCols = 4;

% now finally do the plots
fsize=10;
hfig = figure;
set(hfig,'Units','centimeters','defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
[haxes,axpos]=tightSubplot(nSubplotRows,nSubplotCols,[.02 .02],[.3 .02],[.1 .02]);

axes(haxes(1));
plot(expected_wFluct_averages,zOverL_cc,colormarkers(1),...
    'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),...
    'LineWidth',linewidth(1),'MarkerSize',markersize(1));
hold on;
for idx = 1:nDatasets
    plot(wFluct_averages(idx,:),statisticsGrid_zOverL,colormarkers(idx+1),...
        'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),...
        'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
end
xlabel('$\langle w^\prime \rangle$');
if ~isstring(wFluct_averages_Lim)
    xlim(wFluct_averages_Lim);
end
ylabel("$z/L$");
ylim(z_plotLim);
grid on
hold off;

axes(haxes(2));
plot(expected_wFluct_variances,zOverL_cc,colormarkers(1),...
    'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),...
    'MarkerFaceColor',colormap(1,:),'LineWidth',linewidth(1),'MarkerSize',markersize(1));
hold on;
for idx = 1:nDatasets
    plot(wFluct_variances(idx,:),statisticsGrid_zOverL,colormarkers(idx+1),...
        'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),...
        'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
end
xlabel('$\langle (w^\prime)^2 \rangle$');
if ~isstring(wFluct_variances_Lim)
    xlim(wFluct_variances_Lim);
end
ylabel("$z/L$");
ylim(z_plotLim);
grid on
hold off;

axes(haxes(3));
plot(expected_delta_wFluct_averages,zOverL_cc,colormarkers(1),...
    'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),...
    'LineWidth',linewidth(1),'MarkerSize',markersize(1));
hold on;
for idx = 1:nDatasets
    plot(delta_wFluct_averages(idx,:),statisticsGrid_zOverL,colormarkers(idx+1),...
        'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),...
        'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
end
xlabel('$\langle \Delta w^\prime \rangle/\Delta t$');
if ~isstring(delta_wFluct_averages_Lim)
    xlim(delta_wFluct_averages_Lim);
end
ylabel("$z/L$");
ylim(z_plotLim);
grid on
hold off;

axes(haxes(4));
plot(expected_delta_wFluct_variances,zOverL_cc,colormarkers(1),...
    'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),...
    'LineWidth',linewidth(1),'MarkerSize',markersize(1));
hold on;
for idx = 1:nDatasets
    plot(delta_wFluct_variances(idx,:),statisticsGrid_zOverL,colormarkers(idx+1),...
        'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),...
        'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
end
xlabel('$\langle (\Delta w^\prime)^2 \rangle/\Delta t$');
if ~isstring(delta_wFluct_variances_Lim)
    xlim(delta_wFluct_variances_Lim);
end
ylabel("$z/L$");
ylim(z_plotLim);
grid on
hold off;

set(haxes(2:4),'yLabel',[],'yTickLabel',[])

% now prep the legend
legendString = strings(nDatasets+1,1);
legendString(1) = 'exact solution';
for idx = 1:nDatasets
    legendString(idx+1) = sprintf('%s $\\Delta t = %g$ $T= %g$',...
        caseBaseName,timestep_array(idx),currentTime_array(idx));
end
ledgerend = legend(haxes(1),legendString,'Position',[axpos{1}(1) 0.02 .6 .2]);
set(ledgerend,'FontSize',fsize,'interpreter','latex');


% add the title to the plot (can add labels if needed, don't need it)
%title(t,titleString);
%title(titleString);


% now save the figure as both a pdf and a fig file
currentPlotName = sprintf("%s/%s_statisticPlots",plotOutputDir,caseBaseName);
if show_ghostCells == true
    currentPlotName = sprintf("%s_showGhost",currentPlotName);
end
saveas(hfig,currentPlotName);
save2pdf(hfig,currentPlotName,hfig.Position(3:4),fsize)

A=interp1(zOverL_cc,expected_wFluct_averages,statisticsGrid_zOverL);
B=wFluct_averages(end,:);

rmse = sqrt(mean((A-B).^2));
fprintf("RMSE on mean of wFluct: %f\n",rmse);

A=interp1(zOverL_cc,expected_wFluct_variances,statisticsGrid_zOverL);
B=wFluct_variances(end,:);

rmse = sqrt(mean((A-B).^2));
fprintf("RMSE on variance of wFluct: %f\n",rmse);

A=interp1(zOverL_cc,expected_delta_wFluct_averages,statisticsGrid_zOverL);
B=delta_wFluct_averages(end,:);

rmse = sqrt(mean((A-B).^2));
fprintf("RMSE on mean of wFluct/delta t: %f\n",rmse);

A=interp1(zOverL_cc,expected_delta_wFluct_variances,statisticsGrid_zOverL);
B=delta_wFluct_variances(end,:);

rmse = sqrt(mean((A-B).^2));
fprintf("RMSE on var of wFluct/delta t:: %f\n",rmse);


end