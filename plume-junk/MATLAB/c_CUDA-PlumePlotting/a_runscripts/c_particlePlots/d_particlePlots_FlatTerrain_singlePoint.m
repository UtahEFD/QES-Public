

%%%%%%% standard main function stuff


% clear old stuff
clc;
clear;
close all;
path(pathdef);  % resets the path

% return to current directory
mfile_name          = mfilename('fullpath');
[pathstr,name,ext]  = fileparts(mfile_name);
cd(pathstr);


%%% make sure to grab all functions from all folders
workingDir = pwd;
mainCodeDir = sprintf("%s/../../b_src/c_particlePlotFunctions",workingDir);
sharedFunctionsDir = sprintf("%s/../ff_sharedFunctions",mainCodeDir);

specificFunctionsDir = sprintf("%s/functions",mainCodeDir);
addpath(mainCodeDir,sharedFunctionsDir,  specificFunctionsDir);


%%%%%%% end standard main function stuff


%% FlatTerain singlePoint case


%%% set the input file directory (codeInputDir) as well as the file base names

codeInputDir = sprintf("%s/../../../../../testCases/FlatTerrain/c_plumeOutputs/singlePoint",workingDir);
caseBaseName = "FlatTerrain_singlePoint";
folderFileNames = [
    
    "sim_info.txt";      % 1
    
    sprintf("%s_eulerianData.nc",caseBaseName);    % 2
    sprintf("%s_conc.nc",caseBaseName);            % 3
    sprintf("%s_particleInfo.nc",caseBaseName);    % 4
    
    ];


%%% set the plotOutputDir
plotOutputDir = sprintf("%s/../../../../../testCases/FlatTerrain/e_matlabPlotOutput/singlePoint/c_particlePlots",workingDir);


%%%% now load in the data for the desired case
[doesSimExist,     saveBasename, C_0,timestep, current_time,rogueCount,isNotActiveCount, invarianceTol,velThreshold,    urb_times,urb_x,urb_y,urb_z, urb_u,urb_v,urb_w,turb_sig_x,turb_sig_y,turb_sig_z,turb_txx,turb_txy,turb_txz,turb_tyy,turb_tyz,turb_tzz,turb_epps,turb_tke,eul_dtxxdx,eul_dtxydy,eul_dtxzdz,eul_dtxydx,eul_dtyydy,eul_dtyzdz,eul_dtxzdx,eul_dtyzdy,eul_dtzzdz,eul_flux_div_x,eul_flux_div_y,eul_flux_div_z,   lagrToEul_times,lagrToEul_x,lagrToEul_y,lagrToEul_z, lagrToEul_conc,    lagr_times,lagr_parID,  lagr_xPos_init,lagr_yPos_init,lagr_zPos_init,lagr_tStrt,lagr_sourceIdx,  lagr_xPos,lagr_yPos,lagr_zPos,lagr_uFluct,lagr_vFluct,lagr_wFluct,lagr_delta_uFluct,lagr_delta_vFluct,lagr_delta_wFluct,lagr_isRogue,lagr_isActive] = loadSingleCaseData(codeInputDir,folderFileNames);

nTimes = length(lagr_times);
nx = length(urb_x);
ny = length(urb_y);
nz = length(urb_z);





%%% do a 3D scatter plot of the data
%%% make the plot axis be the size of the domain as given by the urb grid
%%% also plot the initial positions as red x symbols, and the following color as blue

% initialize the figure
fig = figure(1);

% adjust figure size
set(fig,'Units', 'Normalized', 'OuterPosition', [0.35, 0.14, 0.6, 0.7]);


% initialize movie frames data structure
movieFrames(nTimes) = struct('cdata',[],'colormap',[]);
for timeIdx = 1:nTimes
    
    % slow down plot output a hint
    pause(0.5);
    
    % sift data to throw out inactive particles
    isActive_indices = find(lagr_isActive(:,timeIdx) == true);
    current_xPos_init = lagr_xPos_init(isActive_indices,1);
    current_yPos_init = lagr_yPos_init(isActive_indices,1);
    current_zPos_init = lagr_zPos_init(isActive_indices,1);
    current_xPos = lagr_xPos(isActive_indices,timeIdx);
    current_yPos = lagr_yPos(isActive_indices,timeIdx);
    current_zPos = lagr_zPos(isActive_indices,timeIdx);
    
    % now plot the initial positions in red as many x shapes
    scatter3(current_xPos_init,current_yPos_init,current_zPos_init,100,'rx','LineWidth',3);
    
    % now plot the current positions
    hold on;
    scatter3(current_xPos,current_yPos,current_zPos,'+', ...
             'MarkerEdgeColor','b','MarkerFaceColor','b');
    
    % now make sure the axis are the size of the domain
    set(gca,'XLim',[urb_x(1) urb_x(nx)],'YLim',[urb_y(1) urb_y(ny)],'ZLim',[urb_z(1) urb_z(nz)]);
    
    % now add title with the time
    titleString = sprintf('%s\ntime %0.2f',caseBaseName,lagr_times(timeIdx));
    % take out and replace underscore '_' char with blank space ' ' char in titleString caused by caseBaseName
    titleString = strrep(titleString,'_',' ');
    title(titleString);
    
    % now add legend with the source info
    legendString = [
                    "initial locations";
                    sprintf("source %d",lagr_sourceIdx(1));
                    ];
    legend(legendString);
    
    % prepare for next plot
    hold off;
    
    
% % %     %%% save plot output, probably going to do a movie instead    
% % %     % set the output plot name
% % %     currentPlotName = sprintf("%s_particlePositionsBySource_%0.2f",caseBaseName,lagr_times(timeIdx));
% % %     % now replace all decimals in dataName with "o" characters
% % %     currentPlotName = strrep(currentPlotName,'.','o');
% % %     % now set the output plot file path
% % %     currentPlotFile = sprintf("%s/%s.png",plotOutputDir,currentPlotName);
% % %     % now save the figure, once as png, once as a matlab figure
% % %     saveas(fig,currentPlotFile);
% % %     saveas(fig,strrep(currentPlotFile,'.png','.fig'));
    
    
    % save the current frame to the movie frames data structure
    movieFrames(timeIdx) = getframe(fig);
    
end

% create an instance of a movie of the movieFrames in a new figure
%  just to test the movie and see what it looks like
%  just play it back one time
% not needed for making the movie, just useful for seeing
fig = figure(1);
playbackFrequency = 1;
fps = 10;
movie(fig,movieFrames,playbackFrequency,fps);


% make the video writer objects for an avi and a mpg4 video
%  mpg4 only works on Windows and MAC not on LUNIX, hence why both types need created
% first create the filenames and choose your fps for both video types
fps = 5;
movieFilename_avi = sprintf("%s/%s_particlePositionsBySource.avi",plotOutputDir,caseBaseName);
movieFilename_mpeg4 = sprintf("%s/%s_particlePositionsBySource.mp4",plotOutputDir,caseBaseName);

% now make the video writer objects needed for writing the movies out
videoWriteObj_avi = VideoWriter(movieFilename_avi,'Uncompressed AVI');
videoWriteObj_mpeg4 = VideoWriter(movieFilename_mpeg4,'MPEG-4');

% set the frame rate for the output movies
videoWriteObj_avi.FrameRate = fps;
videoWriteObj_mpeg4.FrameRate = fps;

% now write the avi video using the movie frames object and the avi video writer object
open(videoWriteObj_avi);
writeVideo(videoWriteObj_avi,movieFrames);
close(videoWriteObj_avi);

% now write the mpeg4 video using the movie frames object and the mpeg4 video writer object
open(videoWriteObj_mpeg4);
writeVideo(videoWriteObj_mpeg4,movieFrames);
close(videoWriteObj_mpeg4);





%%% do a 3D vector plot of the data
%%% make the plot axis be the size of the domain as given by the urb grid

% initialize the figure
fig = figure(1);

% adjust figure size
set(fig,'Units', 'Normalized', 'OuterPosition', [0.35, 0.14, 0.6, 0.7]);


% set a scaling parameter. Already adjusting the scale by the max - min 
% magnitude of the vectors, but thought it would be handy to have another
% way to adjust it
scaleTheScaleFactor = 0.75;


% initialize movie frames data structure
movieFrames(nTimes) = struct('cdata',[],'colormap',[]);
for timeIdx = 1:nTimes
    
    % slow down plot output a hint
    pause(0.5);
    
    % sift data to throw out inactive particles
    isActive_indices = find(lagr_isActive(:,timeIdx) == true);
    current_xPos = lagr_xPos(isActive_indices,timeIdx);
    current_yPos = lagr_yPos(isActive_indices,timeIdx);
    current_zPos = lagr_zPos(isActive_indices,timeIdx);
    current_uFluct = lagr_uFluct(isActive_indices,timeIdx);
    current_vFluct = lagr_vFluct(isActive_indices,timeIdx);
    current_wFluct = lagr_wFluct(isActive_indices,timeIdx);
    
    
    % set the scaling up size of the vectors, have to play with this for each case
    mags = sqrt( current_uFluct.^2 + current_vFluct.^2 + current_wFluct.^2 );
    quiverScale = double((max(mags)-min(mags)))*scaleTheScaleFactor;
    

    % now plot the current velocity fluctuations
    q = quiver3(current_xPos,current_yPos,current_zPos,current_uFluct,current_vFluct,current_wFluct,quiverScale);
    % now make sure the axis are the size of the domain
    set(gca,'XLim',[urb_x(1) urb_x(nx)],'YLim',[urb_y(1) urb_y(ny)],'ZLim',[urb_z(1) urb_z(nz)]);
    % now add title with the time
    titleString = sprintf('%s\ntime %0.2f',caseBaseName,lagr_times(timeIdx));
    % take out and replace underscore '_' char with blank space ' ' char in titleString caused by caseBaseName
    titleString = strrep(titleString,'_',' ');
    title(titleString);

    
    
    % now change the color of the quiver plot to be colored by velocity
    % magnitudes
    % code borrowed from https://stackoverflow.com/questions/29632430/quiver3-arrow-color-corresponding-to-magnitude
    %// Compute the magnitude of the vectors
    mags = sqrt(sum(cat(2, q.UData(:), q.VData(:), ...
                reshape(q.WData, numel(q.UData), [])).^2, 2));

    %// Get the current colormap
    currentColormap = colormap(gca);

    %// Now determine the color to make each arrow using a colormap
    [~, ~, ind] = histcounts(mags, size(currentColormap, 1));

    %// Now map this to a colormap to get RGB
    cmap = uint8(ind2rgb(ind(:), currentColormap) * 255);
    cmap(:,:,4) = 255;
    cmap = permute(repmat(cmap, [1 3 1]), [2 1 3]);

    %// We repeat each color 3 times (using 1:3 below) because each arrow has 3 vertices
    set(q.Head, ...
        'ColorBinding', 'interpolated', ...
        'ColorData', reshape(cmap(1:3,:,:), [], 4).');   %'

    %// We repeat each color 2 times (using 1:2 below) because each tail has 2 vertices
    set(q.Tail, ...
        'ColorBinding', 'interpolated', ...
        'ColorData', reshape(cmap(1:2,:,:), [], 4).');
    
    % add the colorbar. Not sure if it will be showing the right values,
    % cause the colors seem scaled to go between 0 and 1. Figured I would
    % leave it as the next coder's problem
    colorbar
    
    
% % %     %%% save plot output, probably going to do a movie instead
% % %     % set the output plot name
% % %     outputPlotName = sprintf("%s_particleVelFlucts_%0.2f",caseBaseName,lagr_times(timeIdx));
% % %     % now replace all decimals in dataName with "o" characters
% % %     currentPlotName = strrep(currentPlotName,'.','o');
% % %     % now set the output plot file path
% % %     currentPlotFile = sprintf("%s/%s.png",plotOutputDir,currentPlotName);
% % %     % now save the figure, once as png, once as a matlab figure
% % %     saveas(fig,currentPlotFile);
% % %     saveas(fig,strrep(currentPlotFile,'.png','.fig'));
    

    % save the current frame to the movie frames data structure
    movieFrames(timeIdx) = getframe(fig);
    
end


% make the video writer objects for an avi and a mpg4 video
%  mpg4 only works on Windows and MAC not on LUNIX, hence why both types need created
% first create the filenames and choose your fps for both video types
fps = 5;
movieFilename_avi = sprintf("%s/%s_particleVelFlucts.avi",plotOutputDir,caseBaseName);
movieFilename_mpeg4 = sprintf("%s/%s_particleVelFlucts.mp4",plotOutputDir,caseBaseName);

% now make the video writer objects needed for writing the movies out
videoWriteObj_avi = VideoWriter(movieFilename_avi,'Uncompressed AVI');
videoWriteObj_mpeg4 = VideoWriter(movieFilename_mpeg4,'MPEG-4');

% set the frame rate for the output movies
videoWriteObj_avi.FrameRate = fps;
videoWriteObj_mpeg4.FrameRate = fps;

% now write the avi video using the movie frames object and the avi video writer object
open(videoWriteObj_avi);
writeVideo(videoWriteObj_avi,movieFrames);
close(videoWriteObj_avi);

% now write the mpeg4 video using the movie frames object and the mpeg4 video writer object
open(videoWriteObj_mpeg4);
writeVideo(videoWriteObj_mpeg4,movieFrames);
close(videoWriteObj_mpeg4);





