function plotEulerianVals(codeInputFiles,plotFiles, hozAvg, nonDim, uMeanLim,sigma2Lim,eppsLim, u_tao,del)
    
    
    % now load in data
    [fileExists_array, saveBasename_array,  current_time_array,timestep_array,  C_0_array,nParticles_array,  xCellGrid_array,yCellGrid_array,zCellGrid_array,  uMean_data_array,vMean_data_array,wMean_data_array,sigma2_data_array,epps_data_array,txx_data_array,txy_data_array,txz_data_array,tyy_data_array,tyz_data_array,tzz_data_array,  dtxxdx_data_array,dtxydx_data_array,dtxzdx_data_array,dtxydy_data_array,dtyydy_data_array,dtyzdy_data_array,dtxzdz_data_array,dtyzdz_data_array,dtzzdz_data_array,  flux_div_x_data_array,flux_div_y_data_array,flux_div_z_data_array,  txx_old_array,txy_old_array,txz_old_array,tyy_old_array,tyz_old_array,tzz_old_array,uFluct_old_array,vFluct_old_array,wFluct_old_array,  uFluct_array,vFluct_array,wFluct_array,delta_uFluct_array,delta_vFluct_array,delta_wFluct_array,  rogueCount_array,isActive_array,  xPos_array,yPos_array,zPos_array] = loadCodeOutput(codeInputFiles);

    % get number of files
    nCodeInputFiles = length(fileExists_array);
    
    % take the saveBasename_array and create the plotBasename_array
    [plotBasename_array] = saveToPlotBasename(fileExists_array,saveBasename_array);


    % now do each of the plot types for the input data for each file data
    % note that the data is in cell arrays, so it needs unpacked with each
    % call, and done so on a per file basis.
    for fileIdx = 1:nCodeInputFiles

        if fileExists_array(fileIdx) == true

            %%% plot the input profile
            plotInputProfiles(uMeanLim,sigma2Lim,eppsLim,  cell2mat(xCellGrid_array(fileIdx)),cell2mat(yCellGrid_array(fileIdx)),cell2mat(zCellGrid_array(fileIdx)),  cell2mat(uMean_data_array(fileIdx)),cell2mat(vMean_data_array(fileIdx)),cell2mat(wMean_data_array(fileIdx)),cell2mat(sigma2_data_array(fileIdx)),cell2mat(epps_data_array(fileIdx)),  nonDim,hozAvg,  u_tao,del);
            % figure out the additional filename extension
            if nonDim == false
                plotName = "inputProfiles";
            else
                plotName = "inputProfiles_nonDim";
            end
            currentPlotName = sprintf("%s_%s.png",plotFiles(fileIdx),plotName);
            % get the current figure handle for saving the figure
            fig = gcf;
            saveas(fig,currentPlotName);
            saveas(fig,strrep(currentPlotName,'.png','.fig'));
            % delete current figure. Pause before and after to make sure no errors
            % occur in other processes
            pause(3);
            close(fig);
            pause(3);


            %%% plot the input tensors
            plotInputTensors(cell2mat(xCellGrid_array(fileIdx)),cell2mat(yCellGrid_array(fileIdx)),cell2mat(zCellGrid_array(fileIdx)),  cell2mat(txx_data_array(fileIdx)),cell2mat(txy_data_array(fileIdx)),cell2mat(txz_data_array(fileIdx)),cell2mat(tyy_data_array(fileIdx)),cell2mat(tyz_data_array(fileIdx)),cell2mat(tzz_data_array(fileIdx)),  cell2mat(flux_div_x_data_array(fileIdx)),cell2mat(flux_div_y_data_array(fileIdx)),cell2mat(flux_div_z_data_array(fileIdx)),  hozAvg);
            % figure out the additional filename extension
            plotName = "inputTensors";
            currentPlotName = sprintf("%s_%s.png",plotFiles(fileIdx),plotName);
            % get the current figure handle for saving the figure
            fig = gcf;
            saveas(fig,currentPlotName);
            saveas(fig,strrep(currentPlotName,'.png','.fig'));
            % delete current figure. Pause before and after to make sure no errors
            % occur in other processes
            pause(3);
            close(fig);
            pause(3);

            %%% plot the input tensor derivatives
            plotInputTensorDerivatives(cell2mat(xCellGrid_array(fileIdx)),cell2mat(yCellGrid_array(fileIdx)),cell2mat(zCellGrid_array(fileIdx)),  cell2mat(dtxxdx_data_array(fileIdx)),cell2mat(dtxydx_data_array(fileIdx)),cell2mat(dtxzdx_data_array(fileIdx)),cell2mat(dtxydy_data_array(fileIdx)),cell2mat(dtyydy_data_array(fileIdx)),cell2mat(dtyzdy_data_array(fileIdx)),cell2mat(dtxzdz_data_array(fileIdx)),cell2mat(dtyzdz_data_array(fileIdx)),cell2mat(dtzzdz_data_array(fileIdx)),  hozAvg);
            % figure out the additional filename extension
            plotName = "inputTensorDerivatives";
            currentPlotName = sprintf("%s_%s.png",plotFiles(fileIdx),plotName);
            % get the current figure handle for saving the figure
            fig = gcf;
            saveas(fig,currentPlotName);
            saveas(fig,strrep(currentPlotName,'.png','.fig'));
            % delete current figure. Pause before and after to make sure no errors
            % occur in other processes
            pause(3);
            close(fig);
            pause(3);


        end     % if file exists

    end     % for fileIdx loop
    
end