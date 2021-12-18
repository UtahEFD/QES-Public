function [simVarsExist,  lagrToEul_times,lagrToEul_x,lagrToEul_y,lagrToEul_z, lagrToEul_conc] = readNetcdfLagrToEulFile(lagrToEulFile)

    % not sure of another way to catch if the netcdf read stuff fails, so
    % going to assume any error means that simVars do NOT exist
    try
        
        %%% if desired, get info about the urb file
% %         ncdisp(lagrToEulFile);

        %%% ncdisp shows variables t, x, y, z,  conc
        %%% where t is size 1x1 in dimensions (t) with units 's' and long_name 'time' with datatype double, 
        %%% where x is size 200x1 in dimensions (x) with units 'm' and long_name 'x-distance' with datatype single,
        %%% where y is size 200x1 in dimensions (y) with units 'm' and long_name 'y-distance' with datatype single,
        %%% where z is size 200x1 in dimensions (z) with units 'm' and long_name 'z-distance' with datatype single,
        %%% where conc is size 200x200x200x1 in dimensions (x,y,z,t) with units '#ofPar m-3' and long_name 'concentration' with datatype single.
        %%% t is 999 (0 + tavg). x,y, and z are just linspace from domainStart to domainEnd
        %%% conc goes from 0 to 3.8574
        %%% use max(max(max(data))) or min(min(min(data))) to see such info

        lagrToEul_times = ncread(lagrToEulFile,'t');
        lagrToEul_x = ncread(lagrToEulFile,'x');
        lagrToEul_y = ncread(lagrToEulFile,'y');
        lagrToEul_z = ncread(lagrToEulFile,'z');
        lagrToEul_conc = ncread(lagrToEulFile,'conc');
        
        % got to here without failing the try catch statement, so simVars
        % DO exist
        simVarsExist = true;
        
    catch
        
        simVarsExist = false;
        
        % possible nothing was loaded, so need to set all the output to NAN
        lagrToEul_times = NaN;
        lagrToEul_x = NaN;
        lagrToEul_y = NaN;
        lagrToEul_z = NaN;
        lagrToEul_conc = NaN;
        
    end

end