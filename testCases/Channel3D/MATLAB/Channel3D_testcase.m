% Channel 3D test case for QES-plume
% Base on Bailey 2017 (BLM)
% The channel-flow data: DNS of Kim et al. (1987)and Mansour et al. (1988)
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
nx=51;ny=51;
% grid resolution in z is set to have:
% 50 faces in the plume domain -> 2 extra face on top (above) and bottom (below)
% -> 49 cell within the plume domain -> 2 extra cell on top (above) and bottom (below)
nz=52;

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
uMean_file = 'channel_u.txt';
C0 = 4.0;
epps_file = 'channel_epps.txt';
sigma2_file = 'channel_sigma2.txt';

uMean = load(uMean_file);
zchannel = linspace(0,lz,length(uMean));
epps = load(epps_file);
sigma2 = load(sigma2_file);

%% ========================================================================
% QES-WINDS data:

% face-center data:
u_out = zeros(nx,ny,nz);

% this will return data at k=2...50
u_new=interp1(zchannel,uMean,z_cc);
u_new(1)=-u_new(2); % ghost cell
u_new(end)=u_new(end-1); % ghost cell

for k=1:nz-1
    %u_out(:,:,k)=u_new(k);
    u_out(:,1:ny-1,k)=u_new(k);
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

%icellflag_out(:,:,1) = 2; % terrain 
icellflag_out(:,:,1) = 1; % fluid

% now save the netcdf wind output
writeNetCDFFile_winds('../QES-data/Channel3D',nx,ny,nz,x_cc,y_cc,z_cc,u,v,w,icellflag);


%% ========================================================================
% QES-TURB data:
sig2_new=interp1(zchannel,sigma2,z_cc);
sig2_new(1)=sig2_new(2); % ghost cell
sig2_new(end)=sig2_new(end-1); % ghost cell
epps_new=interp1(zchannel,epps,z_cc);
epps_new(2)=epps(2); % 1st point (to get correct wall dissipation)
epps_new(1)=epps(2); % ghost cell 
epps_new(end)=epps_new(end-1); % ghost cell

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
writeNetCDFFile_turb('../QES-data/Channel3D',x_cc,y_cc,z_cc,CoEps,tke,txx,txy,txz,tyy,tyz,tzz);



