
% set the case base name for use in all the other file paths
caseNameWinds = "PowerLawBLFlow_test";
caseNamePlume = "PowerLawBLFlow_test";
%caseNamePlume = "ContRelease_xDir";

data=struct();
varnames=struct();

% read wind netcdf file
fileName = sprintf("../QES-data/%s_windsOut.nc",caseNameWinds);
[data.winds,varnames.winds] = readNetCDF(fileName);
% read turb netcdf file
fileName = sprintf("../QES-data/%s_turbOut.nc",caseNameWinds);
[data.turb,varnames.turb] = readNetCDF(fileName);

% read main plume files
fileName = sprintf("../QES-data/%s_plumeOut.nc",caseNamePlume);
[data.plume,varnames.plume] = readNetCDF(fileName);

figure
plot(squeeze(data.winds.u(10,10,:)),data.winds.z)
hold all
plot(uPowBL,z_cc)
title('U')

tt_SijSij = data.turb.Sxx .* data.turb.Sxx + data.turb.Syy.* data.turb.Syy + data.turb.Szz .* data.turb.Szz ...
    + 2.0 * (data.turb.Sxy .* data.turb.Sxy + data.turb.Sxz .* data.turb.Sxz + ...
    data.turb.Syz .* data.turb.Syz);
figure
plot(squeeze(tt_SijSij(10,10,:)),data.turb.z)
title('SijSij')

tt_nu_t = data.turb.L .* data.turb.L .* sqrt(2.0 * tt_SijSij);

figure
plot(squeeze(tt_nu_t(10,10,:)),data.turb.z)
title('nu_t')

tmp= sqrt(2.0 * tt_SijSij);
figure
plot(squeeze(tmp(10,10,:)),data.turb.z)


tt_tke = (tt_nu_t ./ (0.55 * data.turb.L)).^(2.0);
tt_CoEps = 5.7 * (sqrt(tt_tke) * 0.55).^(3.0) ./ (data.turb.L);

figure
plot(squeeze(data.turb.L(10,10,:)),data.turb.z)
hold all
plot(0.4*z_cc,z_cc)
title('l_m')

figure
plot(squeeze(data.turb.CoEps(10,10,:)),data.turb.z)
hold all
plot(squeeze(tt_CoEps(10,10,:)),data.turb.z)
plot(squeeze(CoEps(10,10,:)),z_cc)
title('CoEps')

figure
plot(squeeze(data.turb.Sxz(10,10,:)),data.turb.z)
hold all
plot(0.5*dudz,z_cc)
title('Sxz')

figure
plot(squeeze(data.turb.tke(10,10,:)),data.turb.z)
hold all
plot(squeeze(tt_tke(10,10,:)),data.turb.z)
plot(squeeze(tke(10,10,:)),z_cc)
title('tke')

figure
plot(squeeze(data.turb.txz(10,10,:)),data.turb.z)
hold all
plot(squeeze(txz(10,10,:)),z_cc)
title('txz')

figure
plot(squeeze(data.turb.txx(10,10,:)),data.turb.z)
hold all
plot(squeeze(data.turb.tyy(10,10,:)),data.turb.z)
plot(squeeze(data.turb.tzz(10,10,:)),data.turb.z)
plot(squeeze(data.turb.txy(10,10,:)),data.turb.z)
plot(squeeze(data.turb.txz(10,10,:)),data.turb.z)
plot(squeeze(data.turb.tyz(10,10,:)),data.turb.z)

title('tij')