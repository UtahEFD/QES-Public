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

% dimensions of the 3D domain
lx=100;ly=100;lz=20;

% grid resolution in x and y set to have 50 cells
nx=101;ny=101;
% grid resolution in z is set to have:
% 141 faces in the plume domain -> 2 extra face on top (above) and bottom (below)
% -> 140 cell within the plume domain -> 2 extra cell on top (above) and bottom (below)
nz=23;

dx=lx/(nx-1);dy=ly/(ny-1);
dz=lz/(nz-3); 

% grid definition for cell center variables
x_cc=0.5*dx:dx:lx-0.5*dx;
y_cc=0.5*dy:dy:ly-0.5*dy;
% grid definition for cell center variables -> 2 extra cell on top (above) and bottom (below)
z_cc=-0.5*dz:dz:lz+0.5*dz;

% set Power Law BL Flow param:
uRef = 5.90; % m/s
hRef = 4.0; % m
nPow = 0.15; 
uStar = 0.18; % m/s
C0 = 4.0;

%% ========================================================================
% QES-WINDS data:

% face-center data:
u_out = zeros(nx,ny,nz);

uPowBL=uRef*(z_cc/hRef).^nPow;
% BC in the ghost cell:
uPowBL(1)=-uPowBL(2); % ghost cell
uPowBL(end)=uPowBL(end-1); % ghost cell

for k=1:nz-1
    u_out(:,1:ny-1,k)=uPowBL(k);
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
writeNetCDFFile_winds('../QES-data/PowerLawBLFlow',nx,ny,nz,x_cc,y_cc,z_cc,u,v,w,icellflag);


%% ========================================================================
% QES-TURB data:

% cell-center data:
CoEps = zeros(nx-1,ny-1,nz-1);
tke = zeros(nx-1,ny-1,nz-1);

for k=2:nz-1
    CoEps(:,:,k)=C0*uStar^3/0.4/z_cc(k);
    tke(:,:,k)=( z_cc(k)*C0*uStar^3/0.4/z_cc(k) )^(2/3);
end
CoEps(:,:,1)=-CoEps(:,:,2);
tke(:,:,1)=-tke(:,:,2);

% data for NetCDF file
tau11 = (2.0*uStar)^2*ones(nx-1,ny-1,nz-1);
tau22 = (1.6*uStar)^2*ones(nx-1,ny-1,nz-1);
tau33 = (1.3*uStar)^2*ones(nx-1,ny-1,nz-1);

tau13 = (uStar)^2*ones(nx-1,ny-1,nz-1);

tau12 = zeros(nx-1,ny-1,nz-1);
tau23 = zeros(nx-1,ny-1,nz-1);

% now save the netcdf turb output
writeNetCDFFile_turb('../QES-data/PowerLawBLFlow',x_cc,y_cc,z_cc,CoEps,tke,tau11,tau12,tau13,tau22,tau23,tau33);



