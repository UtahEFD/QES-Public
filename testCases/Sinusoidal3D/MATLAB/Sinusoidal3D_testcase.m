% Sinusoidal 3D test case for QES-plume
% Base on Bailey 2017 (BLM)
% 
% Note: the original test case is 1D. This test case has been adapted to 3D
%       the data is repeated in x and y.
%
% F. Margaiaraz
% Univesity of Utah. 2020

%% ========================================================================
% setup:

% dimensions of the 3D domain
lx=1;ly=1;lz=1;

% grid resolution in x and y set to have 50 cells
nx=20+1;ny=20+1;
% grid resolution in z is set to have:
% 50 faces in the plume domain -> 2 extra face on top (above) and bottom (below)
% -> 49 cell within the plume domain -> 2 extra cell on top (above) and bottom (below)
nz=50+2;

% 50 faces 49 cell within the plume domain
dx=lx/(nx-1);dy=ly/(ny-1);
% 50 faces 49 cell within the plume domain
dz=lz/(nz-3); 

% grid definition for cell center variables
x_cc=0.5*dx:dx:lx-0.5*dx;
y_cc=0.5*dy:dy:ly-0.5*dy;
% grid definition for cell center variables -> 2 extra cell on top (above) and bottom (below)
z_cc=-0.5*dz:dz:lz+0.5*dz;

% loading original data set
C0 = 4.0;
zth=linspace(0,lz,101);
uMean=zeros(size(zth));
sigma2=1.1+sin(2*pi*zth);
epps=sigma2.^(3/2);
%% ========================================================================
% QES-WINDS data:

% data for NetCDF file
u = zeros(nx,ny,nz);
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

%icellflag_out(:,:,1) = 2; % terrain 
icellflag_out(:,:,1) = 1; % fluid

% now save the netcdf wind output
writeNetCDFFile_winds('../QES-data/Sinusoidal3D',nx,ny,nz,x_cc,y_cc,z_cc,u,v,w,icellflag);


%% ========================================================================
% QES-TURB data:
sig2_new=1.1 + sin(2*pi*z_cc);
%sig2_new(1)=sig2_new(2); % ghost cell
%sig2_new(end)=sig2_new(end-1); % ghost cell
epps_new=sig2_new.^(3/2);
%epps_new(1)=epps_new(2); % ghost cell 
%epps_new(end)=epps_new(end-1); % ghost cell

% cell-center data:
sig2_out = zeros(nx-1,ny-1,nz-1);
tke_out = zeros(nx-1,ny-1,nz-1);
CoEps_out = zeros(nx-1,ny-1,nz-1);

for k=1:nz-1
    sig2_out(:,:,k)=sig2_new(k);
    CoEps_out(:,:,k)=C0*epps_new(k);
    tke_out(:,:,k)=(z_cc(k)*epps_new(k))^(2/3);
end

% data for NetCDF file
txx = sig2_out;
txy = zeros(nx-1,ny-1,nz-1);
txz = zeros(nx-1,ny-1,nz-1);
tyy = sig2_out;
tyz = zeros(nx-1,ny-1,nz-1);
tzz = sig2_out;
tke = tke_out;
CoEps = CoEps_out;

% now save the netcdf turb output
writeNetCDFFile_turb('../QES-data/Sinusoidal3D',x_cc,y_cc,z_cc,CoEps,tke,txx,txy,txz,tyy,tyz,tzz);

