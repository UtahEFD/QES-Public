function [simVarsExist,  lagr_times,lagr_parID,  lagr_xPos_init,lagr_yPos_init,lagr_zPos_init,lagr_tStrt,lagr_sourceIdx,  lagr_xPos,lagr_yPos,lagr_zPos,lagr_uFluct,lagr_vFluct,lagr_wFluct,lagr_delta_uFluct,lagr_delta_vFluct,lagr_delta_wFluct,lagr_isRogue,lagr_isActive] = readNetcdfLagrangianFile(lagrFile)

    % not sure of another way to catch if the netcdf read stuff fails, so
    % going to assume any error means that simVars do NOT exist
    try
        
        %%% if desired, get info about the urb file
% %         ncdisp(lagrFile);

        %%% ncdisp shows variables t, parID,  xPos_init, yPos_init, zPos_init, tStrt, sourceIdx, 
        %%%  xPos, yPos, zPos, uFluct, vFluct, wFluct, delta_uFluct, delta_vFluct,
        %%%  delta_wFluct, isRogue, isActive
        %%% where t is size 100x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
        %%% where parID is size 100000x1 in dimensions (parID) with units '--' and long_name 'particle ID' with datatype int32,
        %%% where xPos_init is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'initial-x-position' with datatype single,
        %%% where yPos_init is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'initial-y-position' with datatype single,
        %%% where zPos_init is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'initial-z-position' with datatype single,
        %%% where tStrt is size 100000x100 in dimensions (parID,t) with units 's' and long_name 'particle-release-time' with datatype single,
        %%% where sourceIdx is size 100000x100 in dimensions (parID,t) with units '--' and long_name 'particle-sourceID' with datatype int32,
        %%% where xPos is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'x-position' with datatype single,
        %%% where yPos is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'y-position' with datatype single,
        %%% where zPos is size 100000x100 in dimensions (parID,t) with units 'm' and long_name 'z-position' with datatype single,
        %%% where uFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'u-velocity-fluctuation' with datatype single,
        %%% where vFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'v-velocity-fluctuation' with datatype single,
        %%% where wFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'w-velocity-fluctuation' with datatype single,
        %%% where delta_uFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'uFluct-difference' with datatype single,
        %%% where delta_vFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'vFluct-difference' with datatype single,
        %%% where delta_wFluct is size 100000x100 in dimensions (parID,t) with units 'm s-1' and long_name 'wFluct-difference' with datatype single,
        %%% where isRogue is size 100000x100 in dimensions (parID,t) with units 'bool' and long_name 'is-particle-rogue' with datatype int32,
        %%% where isActive is size 100000x100 in dimensions (parID,t) with units 'bool' and long_name 'is-particle-rogue' with datatype int32.
        %%% t is a list from 0 to 990 in increments of 10.

        lagr_times = ncread(lagrFile,'t');
        lagr_parID = ncread(lagrFile,'parID');
        lagr_xPos_init = ncread(lagrFile,'xPos_init');
        lagr_yPos_init = ncread(lagrFile,'yPos_init');
        lagr_zPos_init = ncread(lagrFile,'zPos_init');
        lagr_tStrt = ncread(lagrFile,'tStrt');
        lagr_sourceIdx = ncread(lagrFile,'sourceIdx');
        lagr_xPos = ncread(lagrFile,'xPos');
        lagr_yPos = ncread(lagrFile,'yPos');
        lagr_zPos = ncread(lagrFile,'zPos');
        lagr_uFluct = ncread(lagrFile,'uFluct');
        lagr_vFluct = ncread(lagrFile,'vFluct');
        lagr_wFluct = ncread(lagrFile,'wFluct');
        lagr_delta_uFluct = ncread(lagrFile,'delta_uFluct');
        lagr_delta_vFluct = ncread(lagrFile,'delta_vFluct');
        lagr_delta_wFluct = ncread(lagrFile,'delta_wFluct');
        lagr_isRogue = ncread(lagrFile,'isRogue');
        lagr_isActive = ncread(lagrFile,'isActive');
        
        % got to here without failing the try catch statement, so simVars
        % DO exist
        simVarsExist = true;
        
    catch
        
        simVarsExist = false;
        
        % possible nothing was loaded, so need to set all the output to NAN
        lagr_times = NaN;
        lagr_parID = NaN;
        lagr_xPos_init = NaN;
        lagr_yPos_init = NaN;
        lagr_zPos_init = NaN;
        lagr_tStrt = NaN;
        lagr_sourceIdx = NaN;
        lagr_xPos = NaN;
        lagr_yPos = NaN;
        lagr_zPos = NaN;
        lagr_uFluct = NaN;
        lagr_vFluct = NaN;
        lagr_wFluct = NaN;
        lagr_delta_uFluct = NaN;
        lagr_delta_vFluct = NaN;
        lagr_delta_wFluct = NaN;
        lagr_isRogue = NaN;
        lagr_isActive = NaN;
        
    end

end