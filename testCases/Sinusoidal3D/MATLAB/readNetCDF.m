function [data,v] = readNetCDF(filename)

%ncdisp(filename)

% open NetCDF file
ncid = netcdf.open(filename,'NC_NOWRITE');
netcdf.close(ncid);

% get variables name
finfo=ncinfo(filename);
varname={finfo.Variables.Name};

% read variables name
for j=1:numel(varname)
    v=varname{j};
    data.(v)=ncread(filename,v);
end

end