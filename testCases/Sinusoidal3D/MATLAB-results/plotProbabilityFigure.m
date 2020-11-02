function plotProbabilityFigure( caseBaseName,plotOutputDir,  probabilityLim, nBins,  currentTime_array,timestep_array,  Lz,  caseNames, plumeParInfo_data )

% get the number of datasets to plot
nDatasets = length(caseNames);

% pull out some of the data from the datasets
for fileIdx = 1:nDatasets
    
    % grab all par data from the very last timeset, nothing else
    % data is in format (nPar,nTimes) for each dataset in a case
    %isRogue_array.(caseNames(fileIdx)) = plumeParInfo_data.(caseNames(fileIdx)).isRogue(:,end);
    %isActive_array.(caseNames(fileIdx)) = plumeParInfo_data.(caseNames(fileIdx)).isActive(:,end);
    
    zPos_array.(caseNames(fileIdx)) = plumeParInfo_data.(caseNames(fileIdx)).zPos(:,end);
    
end


% create the edge grid, to make histcounts easier and more correct
nEdges = nBins+1;   % edges are always one index more, cause it is the faces of the bins
gridEdges_zOverL = linspace(0,1,nEdges);



% set the subplot size
nSubplotRows = 1;
nSubplotCols = nDatasets;


% set the title string
titleString = sprintf('Resulting Probability Profiles');


% now finally do the plots
fsize=10;
hfig = figure;
set(hfig,'Units','centimeters','defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
[haxes,axpos]=tightSubplot(nSubplotRows,nSubplotCols,[.02 .02],[.3 .02],[.1 .02]);

for idx = 1:nDatasets
    %for idx = nDatasets:-1:1   % reverse the order in the plot
    
    % first get the current dataset data
    current_currentTime = currentTime_array(idx);
    current_timestep = timestep_array(idx);
    
    %current_isRogue = isRogue_array.(caseNames(idx));
    %current_isActive = isActive_array.(caseNames(idx));
    
    current_zPos = zPos_array.(caseNames(idx));
    
    
    % get the total number of particles
    nPar = length(current_zPos);
    
    % now get the indices for which particles are active
    %current_isActive_indices = find( current_isActive == true );
    
    % calculate the number of active particles
    %nActivePar = length(current_isActive_indices);
    nActivePar = nPar;
    
    % calculate the number of rogue particles
    %current_isRogue_indices = find( current_isRogue == true );
    nRoguePar = 0; %length(current_isRogue_indices);
    
    
    % now filter the current dataset data by active particles
    % (assumes rogue particles are already set to inactive)
    currentActive_zPos = current_zPos;
    
    % correct the particle position values to be zOverL
    currentActive_zPosOverL = currentActive_zPos/Lz;
    
    
    %%% now calculate the probability information
    
    % first count the number of active particles found in each bin
    current_parsPerBin = histcounts(currentActive_zPosOverL,gridEdges_zOverL);
    
    % now calculate the expected number of particles, adjusting for
    % inactive and rogue particles being thrown out
    current_expectedParsPerBin_corrected = nActivePar/nBins;
    
    % now calculate the probability, entropy, and rogue ratio values
    current_probabilities = current_parsPerBin/current_expectedParsPerBin_corrected;
    current_entropy = -sum(current_probabilities(:).*log(current_probabilities(:)));
    current_rogueRatio = nRoguePar/nPar;    % this is the total number of par including rogue and inactive particles
    
    
    % the expected probability for this case is always 1. The desired
    % result is that found parsPerBin matches expectedParsPerBin, which
    % only happens if parsPerBin matches idealParsPerBin.
    % idealParsPerBin = nPar/nBins
    % expectedParsPerBin = nPar/nBins  ( so equals idealParsPerBin no matter what )
    % expectedProbability = idealParsPerBin/expectedParticlesPerBin
    % note that this still holds true even if you correct
    %  idealParsPerBin and expectedParsPerBin by throwing out inactive particles:
    % idealParsPerBin_corrected = nActivePar/nBins
    % expectedParsPerBin_corrected = nActivePar/nBins
    % expectedProbability = idealParsPerBin_corrected/expectedParsPerBin_corrected
    %  the difference between the expected/ideal and what is seen is that
    %  the probability is calculated differently than expectedProbability
    % probabilities = foundParsPerBin/expectedParsPerBin
    % expectedProbabilities = idealParsPerBin/expectedParsPerBin
    %  this holds true when correcting for thrown out inactive particles too
    expectedProbability = 1;
    
    
    axes(haxes(idx));
    
    histogram('BinCounts',current_probabilities,'BinEdges',gridEdges_zOverL,'Orientation','horizontal');
    hold on;
    plot([expectedProbability expectedProbability],[0 1],'k--');
    xlabel(sprintf(['Probability for \n%s\n $\\Delta t = %g$ $T = %g$.\n',...
        '$%d/%d$ rogue\n rogue ratio $%g$\n entropy $S=%.3g$'],...
        caseBaseName,current_timestep,current_currentTime,...
        nRoguePar,nPar,current_rogueRatio,current_entropy));
    if ~isstring(probabilityLim)
        xlim(probabilityLim);
    end
    ylabel("$z/L$");
    ylim([0 1]);
    hold off;
    grid on
    
end     % for idx

% add the title to the plot (can add labels if needed, don't need it)
%title(t,titleString);
set(haxes(2:end),'yLabel',[],'yTickLabel',[])


% adjust figure size
% might have to play with the position vector, is used to improve
% margin whitespace done by tilelayout, but this means it can also undo
% the improvements made to the margin whitespace by tilelayout.
%fighandles = findall( allchild(0), 'type', 'figure');
%%figPosition = get(fighandles(1),'position');
%set(fighandles(1),'Units', 'Normalized', 'OuterPosition', [0.2, 0.1, 0.57, 0.85]);


% now save the figure as both a pdf and a fig file
currentPlotName = sprintf("%s/%s_probabilityPlots",plotOutputDir,caseBaseName);
saveas(hfig,currentPlotName);
save2pdf(hfig,currentPlotName,hfig.Position(3:4),fsize)



end