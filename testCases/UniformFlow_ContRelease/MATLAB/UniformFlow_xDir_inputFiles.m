 
%% ========================================================================
% setup:

%file name to save netCDF (must contain path)
filename='../QES-data/UniformFlow_xDir';

% dimensions of the 3D domain
lx=100;ly=100;lz=140;

% grid resolution in x and y set to have 50 cells
nx=103;ny=103;
% grid resolution in z is set to have:
% 141 faces in the plume domain -> 2 extra face on top (above) and bottom (below)
% -> 140 cell within the plume domain -> 2 extra cell on top (above) and bottom (below)
nz=143;

dx=lx/(nx-3);dy=ly/(ny-3);
dz=lz/(nz-3); 

% grid definition for cell center variables
x_cc=-0.5*dx:dx:lx+0.5*dx;
y_cc=-0.5*dy:dy:ly+0.5*dy;
% grid definition for cell center variables -> 2 extra cell on top (above) and bottom (below)
z_cc=-0.5*dz:dz:lz+0.5*dz;

% set Uniform Flow param:
uMean = 2.0; % m/s
uStar = 0.174; % m/s
C0 = 4.0;

%% ========================================================================
% QES-WINDS data:

% face-center data:
u_out = zeros(nx,ny,nz);

% this will return data at k=2...50
u_new=uMean*ones(nz-1,1);
% BC in the ghost cell:
%u_new(1)=u_new(2); % ghost cell
%u_new(end)=u_new(end-1); % ghost cell

for kk=1:nz-1
    u_out(:,1:ny-1,kk)=u_new(kk);
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

k = (uStar/0.55)^2;

sigU=2.50*uStar;
sigV=1.78*uStar;
sigW=1.27*uStar;

% cell-center data:
CoEps = zeros(nx-1,ny-1,nz-1);
tke = zeros(nx-1,ny-1,nz-1);

for kk=2:nz-1
    CoEps(:,:,kk)=5.7*(sqrt(k)*0.55)^3/(0.4*z_cc(kk));
    tke(:,:,kk)=k;
end
CoEps(:,:,1)=-CoEps(:,:,2);
tke(:,:,1)=-tke(:,:,2);

% data for NetCDF file
txx = sigU^2 * ones(nx-1,ny-1,nz-1);
tyy = sigV^2 * ones(nx-1,ny-1,nz-1);
tzz = sigW^2 * ones(nx-1,ny-1,nz-1);

txz = -(uStar)^2*ones(nx-1,ny-1,nz-1);

txy = zeros(nx-1,ny-1,nz-1);
tyz = zeros(nx-1,ny-1,nz-1);

% now save the netcdf turb output
writeNetCDFFile_turb(filename,x_cc,y_cc,z_cc,CoEps,tke,txx,txy,txz,tyy,tyz,tzz);



