% Power Law Bound. Layer Flow test case for QES-plume
% Base on Singh PhD Dissertation )
% Initial test case published in 
%  Singh et al. 2004 
%  Willemsen et al. 2007
%
% F. Margaiaraz
% Univesity of Utah. 2021

%% ========================================================================
% setup:

%file name to save netCDF (must contain path)
filename='../QES-data/PowerLawBLFlow';

% dimensions of the 3D domain
lx=100;ly=100;lz=20;

% grid resolution in x and y set to have 50 cells
nx=101;ny=101;
% grid resolution in z is set to have:
% 141 faces in the plume domain -> 2 extra face on top (above) and bottom (below)
% -> 140 cell within the plume domain -> 2 extra cell on top (above) and bottom (below)
nz=43;

dx=lx/(nx-1);dy=ly/(ny-1);
dz=lz/(nz-3); 

% grid definition for cell center variables
x_cc=0.5*dx:dx:lx-0.5*dx;
y_cc=0.5*dy:dy:ly-0.5*dy;
% grid definition for cell center variables -> 2 extra cell on top (above) and bottom (below)
z_cc=-0.5*dz:dz:lz+0.5*dz; z_cc=z_cc';

% set Power Law BL Flow param:
a = 4.8; % m^(1-p)/s
p = 0.15; 
b = 0.08;
n = 1;

uRef = 5.90; % m/s
hRef = 4.0; % m
uStar = 0.20; % m/s
C0 = 4.0;

%% ========================================================================
% QES-WINDS data:

% face-center data:
u_out = zeros(nx,ny,nz);

uPowBL=a*(z_cc).^p;
% BC in the ghost cell:
uPowBL(1)=-uPowBL(2); % ghost cell
uPowBL(end)=uPowBL(end-1); % ghost cell

for kk=1:nz-1
    u_out(:,1:ny-1,kk)=uPowBL(kk);
end

% data for NetCDF file
u = u_out; 
v = zeros(nx,ny,nz);
w = zeros(nx,ny,nz);

% cell-center data:
icellflag_out = ones(nx-1,ny-1,nz-1);

icellflag = icellflag_out;
% this can be used to check terrain reflection versus domain BC reflection:
% - if icellflag is set to 2 (terrain) and reflection is enable 
%   -> QES-plume will perform a trajectory reflection for each particle
% - if icellflag is set to 1 (fluid) and BC reflection is enable
%   -> QES-plume will place particle outside the domain inside based on the
%      miror condition
% note: top ghost cell set as fluid: icellflag_out(:,:,nz-1) = 1;

icellflag_out(:,:,1) = 2; % terrain 
%icellflag_out(:,:,1) = 1; % fluid

% now save the netcdf wind output
writeNetCDFFile_winds(filename,nx,ny,nz,x_cc,y_cc,z_cc,u,v,w,icellflag);


%% ========================================================================
% QES-TURB data:

%dudz = zeros(nz-1,1);
%for kk=2:nz-2
%    dudz(kk) = (uPowBL(kk+1)-uPowBL(kk-1))/(2*dz);
%end
%dudz(1) = -dudz(2);

dudz=p*a*z_cc.^(p-1);

ustar = 0.4*z_cc.*dudz;
%ustar = 0.4*p*a*z_cc.^p;
ustar(1) = -ustar(2);
ustar(end) = ustar(end-1);

uw = -b*z_cc.^n.*dudz;
uw(1) = -uw(2);
uw(end) = uw(end-1);

%sigU = 2.30*ustar;
%sigV = 1.80*ustar;
%sigW = 1.00*ustar;

sigU = 2.50*ustar;
sigV = 1.78*ustar;
sigW = 1.27*ustar;

k = (ustar/0.55).^2;
k(1) = -k(2);
k(end) = k(end-1);

nu = 0.4.*z_cc.*ustar;
nu(1) = -nu(2);

eps = 5.7*(sqrt(k)*0.55).^3./(0.4*z_cc);
eps(1) = -eps(2);
eps(end) = eps(end-1);

% stress tensor
txx = zeros(nx-1,ny-1,nz-1);
txz = zeros(nx-1,ny-1,nz-1);
txy = zeros(nx-1,ny-1,nz-1);
tyy = zeros(nx-1,ny-1,nz-1);
tyz = zeros(nx-1,ny-1,nz-1);
tzz = zeros(nx-1,ny-1,nz-1);

for kk=2:nz-1
    txx(:,:,kk) = sigU(kk)^2;%2.0/3.0*k(kk) * (2.0*0.55)^2;
    tyy(:,:,kk) = sigV(kk)^2;%2.0/3.0*k(kk) * (2.0*0.55)^2;
    tzz(:,:,kk) = sigW(kk)^2;%2.0/3.0*k(kk) * (1.3*0.55)^2;
end
txx(:,:,1) = -txx(:,:,2);
tyy(:,:,1) = -tyy(:,:,2);
tzz(:,:,1) = -tzz(:,:,2);

for kk=2:nz-1
    txz(:,:,kk) = -ustar(kk)^2;%nu(kk)*dudz(kk);
end
txz(:,:,1) = -txz(:,:,2);

CoEps = zeros(nx-1,ny-1,nz-1);
tke = zeros(nx-1,ny-1,nz-1);

for kk=1:nz-1
    tke(:,:,kk)=k(kk);
    %tke(:,:,k)=( z_cc(k)*C0*uStar^3/0.4/z_cc(k) )^(2/3);
    CoEps(:,:,kk)=eps(kk);
    %CoEps(:,:,k)=C0*uStar^3/0.4/z_cc(k);
end

% now save the netcdf turb output
writeNetCDFFile_turb(filename,x_cc,y_cc,z_cc,CoEps,tke,txx,txy,txz,tyy,tyz,tzz);


