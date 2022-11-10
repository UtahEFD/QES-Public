function plotStatisticsFigure_LES(caseBaseName,plotOutputDir,...
    zLim,subfilter_tke_Lim,uFluct_wFluct_covariances_Lim,delta_wFluct_variances_Lim,...
    nBins,currentTime_array,timestep_array,zGridInfo,turb_data,caseNames,plumeParInfo_data,C0)

%%% before anything else, pull out the data needed from the packed dataset structures

% pull out the eulerian grid values
zOverL_cc = zGridInfo.zOverL_cc;
Lz = zGridInfo.Lz;


% now pull out the turb data
txz = turb_data.txz;
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
    
    uFluct_array.(caseNames(fileIdx)) = plumeParInfo_data.(caseNames(fileIdx)).uFluct(:,end);
    wFluct_array.(caseNames(fileIdx)) = plumeParInfo_data.(caseNames(fileIdx)).wFluct(:,end);
    delta_wFluct_array.(caseNames(fileIdx)) = plumeParInfo_data.(caseNames(fileIdx)).delta_wFluct(:,end);
    
end


%%% first, calculate the expected values
%%% this includes taking the eulerian data and horizontally averaging it


% now calculate the horizontally averaged eularian values
[ txz_hozAvged ] = hozAvg3D( txz );
[ tzz_hozAvged ] = hozAvg3D( tzz );
[ epps_hozAvged ] = hozAvg3D( epps );

% now use all the above to set the expected particle statistic values
%expected_subfilter_tke = tzz_hozAvged.^(2/3);   
% expect it to be the variance, though maybe it is the variance^2/3.
expected_subfilter_tke = tzz_hozAvged;   % expect it to be the variance, though maybe it is the variance^2/3.
expected_uFluct_wFluct_covariances = txz_hozAvged;   % expect it to be the covariance
% expect it to be C0*epps*timestep, after "dividing" by timestep on paper it's this
expected_delta_wFluct_variances = C0*epps_hozAvged;   



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
subfilter_tke = nan(nDatasets,nBins);
uFluct_wFluct_covariances = nan(nDatasets,nBins);
delta_wFluct_variances = nan(nDatasets,nBins);

% now calculate the particle statistics
for idx = 1:nDatasets
    
    % first get the current dataset data
    current_timestep = timestep_array(idx);
    
    current_isActive = isActive_array.(caseNames(idx));
    
    current_zPos = zPos_array.(caseNames(idx));
    current_uFluct = uFluct_array.(caseNames(idx));
    current_wFluct = wFluct_array.(caseNames(idx));
    current_delta_wFluct = delta_wFluct_array.(caseNames(idx));
    
    
    % now get the indices for which particles are active
    current_isActive_indices = find( current_isActive == true );
    
    
    % now filter the current dataset data by active particles
    % (assumes rogue particles are already set to inactive)
    currentActive_zPos = current_zPos(current_isActive_indices);
    currentActive_uFluct = current_uFluct(current_isActive_indices);
    currentActive_wFluct = current_wFluct(current_isActive_indices);
    currentActive_delta_wFluct = current_delta_wFluct(current_isActive_indices);
    
    % correct the particle position values to be zOverL
    currentActive_zPosOverL = currentActive_zPos/Lz;
    
    % get the indices of a given variable
    [current_N,current_edges,current_bin_indices] = histcounts(currentActive_zPosOverL,gridEdges_zOverL);
    
    
    %%% now calculate the required statistics
    for binIdx = 1:nBins
        %subfilter_tke(idx,binIdx) = var(currentActive_wFluct(current_bin_indices == binIdx)).^(2/3);
        subfilter_tke(idx,binIdx) = var(currentActive_wFluct(current_bin_indices == binIdx));
        currentCovariances = cov(currentActive_uFluct(current_bin_indices == binIdx),...
            currentActive_wFluct(current_bin_indices == binIdx));
        % 1,2 comes from matlab help describing the output as covar(A,A), covar(A,B), covar(B,A), covar(B,B). 
        % This should be covar(A,B).
        uFluct_wFluct_covariances(idx,binIdx) = currentCovariances(1,2);    
        % is divided by timestep on paper to get this eqn
        delta_wFluct_variances(idx,binIdx) = ...
            var(currentActive_delta_wFluct(current_bin_indices == binIdx))/current_timestep;   
    end
    
    
end



%%% now plot the expected values and the found particle statistics

% set the title string
titleString = sprintf('Resulting Statistic Profiles -- LES');

% set some useful plot variables, for controlling the plot a bit
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
    15,
    3,
    4,
    3,
    3
    ];

% set the subplot size
nSubplotRows = 1;
nSubplotCols = 3;

% now finally do the plots
fsize=10;
hfig = figure;
set(hfig,'Units','centimeters','defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
[haxes,axpos]=tightSubplot(nSubplotRows,nSubplotCols,[.02 .02],[.3 .02],[.1 .02]);

axes(haxes(1))
plot(expected_subfilter_tke,zOverL_cc,colormarkers(1),...
    'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),...
    'LineWidth',linewidth(1),'MarkerSize',markersize(1));
hold on;
for idx = 1:nDatasets
    plot(subfilter_tke(idx,:),statisticsGrid_zOverL,colormarkers(idx+1),...
        'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),...
        'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
end
xlabel("$\langle k^2\rangle$");
if ~isstring(subfilter_tke_Lim)
    xlim(subfilter_tke_Lim);
end
ylabel("$z/L$");
if ~isstring(zLim)
    ylim(zLim);
end
grid on
hold off;

axes(haxes(2))
plot(expected_uFluct_wFluct_covariances,zOverL_cc,colormarkers(1),...
    'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),...
    'LineWidth',linewidth(1),'MarkerSize',markersize(1));
hold on;
for idx = 1:nDatasets
    plot(uFluct_wFluct_covariances(idx,:),statisticsGrid_zOverL,colormarkers(idx+1),...
        'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),...
        'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
end
xlabel('$\langle u^\prime w^\prime \rangle$')
if ~isstring(uFluct_wFluct_covariances_Lim)
    xlim(uFluct_wFluct_covariances_Lim);
end
ylabel("$z/L$");
if ~isstring(zLim)
    ylim(zLim);
end
grid on
hold off;

axes(haxes(3))
plot(expected_delta_wFluct_variances,zOverL_cc,colormarkers(1),...
    'Color',colormap(1,:),'MarkerEdgeColor',colormap(1,:),'MarkerFaceColor',colormap(1,:),...
    'LineWidth',linewidth(1),'MarkerSize',markersize(1));
hold on;
for idx = 1:nDatasets
    plot(delta_wFluct_variances(idx,:),statisticsGrid_zOverL,colormarkers(idx+1),...
        'Color',colormap(idx+1,:),'MarkerEdgeColor',colormap(idx+1,:),'MarkerFaceColor',colormap(idx+1,:),...
        'LineWidth',linewidth(idx+1),'MarkerSize',markersize(idx+1));
end
xlabel('$\langle \Delta w^\prime \rangle/\Delta t$');
if ~isstring(delta_wFluct_variances_Lim)
    xlim(delta_wFluct_variances_Lim);
end
ylabel("$z/L$");
if ~isstring(zLim)
    ylim(zLim);
end
grid on
hold off;

set(haxes(2:end),'yLabel',[],'yTickLabel',[])

% now prep the legend
legendString = strings(nDatasets+1,1);
legendString(1) = 'exact solution';
for idx = 1:nDatasets
    legendString(idx+1) = sprintf('%s $\\Delta t = %g$  $T = %g$',...
        caseBaseName,timestep_array(idx),currentTime_array(idx));
end
ledgerend = legend(haxes(1),legendString,'Position',[axpos{1}(1) 0.02 .6 .2]);
set(ledgerend,'FontSize',fsize,'interpreter','latex');

% add the title to the plot (can add labels if needed, don't need it)
%title(t,titleString);


% now save the figure as both a png and a fig file
currentPlotName = sprintf("%s/%s_statisticPlots_LES",plotOutputDir,caseBaseName);
saveas(hfig,currentPlotName);
save2pdf(hfig,currentPlotName,hfig.Position(3:4),fsize)




end