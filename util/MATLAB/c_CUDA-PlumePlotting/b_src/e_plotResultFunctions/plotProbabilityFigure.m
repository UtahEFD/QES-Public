function plotProbabilityFigure(nSubPlotRows,nSubPlotCols,subPlotIndices,  figTitle,dimName,probabilityLim,nParticleBins,  fileExists_array,  plotBasename_array,   current_time_array,timestep_array,nParticles_array,  current_rogueCount_array,current_isActive_array,  current_pos_array)

    %%% note that this expects cell arrays as inputs instead of numeric
    %%% arrays, as it does the conversion inside here.
    %%% the main reason for this, is that cell2mat conglomerates multiple
    %%% array dimensions onto each other if trying to run cell2mat on a set
    %%% of cell values instead of a single cell value.
    
    % make sure input nSubplotRows and nSubplotCols are single values that
    % are greater than zero
    if ~(size(nSubPlotRows,1) == 1 && size(nSubPlotRows,2) == 1) || length(size(nSubPlotRows)) > 2
        error('!!! plotProbabilityFigure error !!! input nSubPlotRows is not a single value !');
    end
    if nSubPlotRows < 1
        error('!!! plotProbabilityFigure error !!! input nSubPlotRows is not greater than zero !');
    end
    if ~(size(nSubPlotCols,1) == 1 && size(nSubPlotCols,2) == 1) || length(size(nSubPlotCols)) > 2
        error('!!! plotProbabilityFigure error !!! input nSubPlotRows is not a single value !');
    end
    if nSubPlotCols < 1
        error('!!! plotProbabilityFigure error !!! input nSubPlotCols is not greater than zero !');
    end
    
    % now make sure subPlotIndices is a row or a column vector
    if (size(subPlotIndices,1) ~= 1 && size(subPlotIndices,2) ~= 1) || length(size(subPlotIndices)) > 2
        error('!!! plotProbabilityFigure error !!! input subPlotIndices is not a row or a column vector !');
    end
    
    % calculate the number of subPlotIndices and make sure it matches the
    % size of the input file array
    nSubPlots = length(subPlotIndices);
    if nSubPlots > nSubPlotRows*nSubPlotCols
        error('!!! plotProbabilityFigure error !!! input subPlotIndices has greater length than nSubPlotRows*nSubPlotCols !');
    end
    if nSubPlots ~= length(plotBasename_array)
        error('!!! plotProbabilityFigure error !!! input subPlotIndices does not have the same number of values as input plotBasename_array !');
    end


    if ~(dimName == "x" || dimName == "y" || dimName == "z")
        error('!!! plotProbabilityFigure error !!! input dimName is not \"x\", \"y\", or \"z\" but is \"%s\" !!!',dimName);
    end
    
    % if probabilityLim is "", then ignore its use. If it isn't it had
    % better be a proper limit. double and string makes matlab mad, so if
    % it is a string, assume it is "". Basically an isnan idea.
    if ~isstring(probabilityLim)
        if size(probabilityLim,1) ~= 1 && size(probabilityLim,2) ~= 1
            error('!!! plotProbabilityFigure error !!! input probabilityLim is not a row or column vector !');
        end
        if length(probabilityLim) ~= 2
            error('!!! plotProbabilityFigure error !!! input probabilityLim is not size 2 !');
        end
        if probabilityLim(2) <= probabilityLim(1)
            error('!!! plotProbabilityFigure error !!! input probabilityLim 2nd posLim is not greater than 1st posLim !');
        end
    end
    
    
    % create the edge grid
    nEdges = nParticleBins+1;   % edges are always one index more, cause it is the faces of the bins
    gridEdges_posOverL = linspace(0,1,nEdges);
    
    
    figure;
    
    for idx = 1:nSubPlots
        
        if fileExists_array(idx) == true
            % notice that it is current_ as opposed to old_ values
            current_isActive_indices = find(cell2mat(current_isActive_array(idx)) == true);
            if ~(isempty(current_isActive_indices))

                % grab the required values for plotting
                % first pull out cell array stuff
                current_time = cell2mat(current_time_array(idx));
                timestep = cell2mat(timestep_array(idx));
                plotBasename = string(plotBasename_array(idx));
                
                current_pos = cell2mat(current_pos_array(idx));
                nParticles = cell2mat(nParticles_array(idx));
                current_rogueCount = cell2mat(current_rogueCount_array(idx));

                % now pull out the required indices of values
                current_pos_current_isActive = current_pos(current_isActive_indices);
                
                % correct the grid to be posOverL if isn't already posOverL
                [current_posOverL_current_isActive,z_nCells] = makePosOverL(current_pos_current_isActive);

                % now do intermediate calculations with the values
                current_particlesPerBin = histcounts(current_posOverL_current_isActive,gridEdges_posOverL);
                current_particlesLeft = nParticles - current_rogueCount;
                current_expectedParticlesPerBin = current_particlesLeft/nParticleBins;  % correct the expected particles per bin
                current_probabilities = current_particlesPerBin/current_expectedParticlesPerBin;
                current_entropy = -sum(current_probabilities(:).*log(current_probabilities(:)));
                current_rogueRatio = current_rogueCount/nParticles;

                expectedProbabilityPerBin = 1;

                subplot(nSubPlotRows,nSubPlotCols,subPlotIndices(idx));
                    histogram('BinCounts',current_probabilities,'BinEdges',gridEdges_posOverL,'Orientation','horizontal');
                    hold on;
                    plot([expectedProbabilityPerBin expectedProbabilityPerBin],[0 1],'k--');
                    xlabel(sprintf('Probability\ntime %0.4f timestep %0.4f\n%s\n%d out of %d particles rogue\n entropy %0.5f rogue ratio %0.3f',current_time,timestep,plotBasename,current_rogueCount,nParticles,current_entropy,current_rogueRatio));
                    if ~isstring(probabilityLim)
                        xlim(probabilityLim);
                    end
                    ylabel(sprintf("%s\\\\L",dimName));
                    ylim([0 1]);
                    hold off;

            end     % if isActive
        
        end     % if file exists
        
    end     % for loop
    
    sgtitle(figTitle);
    drawnow
    % adjust figure size
    fighandles = findall( allchild(0), 'type', 'figure');
    %%figPosition = get(fighandles(1),'position');
    set(fighandles(1),'Units', 'Normalized', 'OuterPosition', [0.1, 0.1, 0.8, 0.85]);
    
end