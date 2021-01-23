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
a=4.8; % m^(1-p)/s
uRef = 5.90; % m/s
hRef = 4.0; % m
nPow = 0.15; 
uStar = 0.20; % m/s
C0 = 4.0;

%% ========================================================================
% QES-WINDS data:

% face-center data:
u_out = zeros(nx,ny,nz);

uPowBL=a*(z_cc).^nPow;
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

dudz = zeros(nz-1,1);
for kk=2:nz-2
    dudz(kk) = (uPowBL(kk+1)-uPowBL(kk-1))/(2*dz);
end
dudz(1) = -dudz(2);

k = 0.4*z_cc.*dudz/0.55;
k(1) = -k(2);
k(end) = k(end-1);

nu = ((0.4*z_cc).^2).*(dudz);
nu(1) = -nu(2);
nu(end) = nu(end-1);

eps = 0.55*k.^(3/2)./(0.4*z_cc); 
eps(1) = -eps(2);
eps(end) = eps(end-1);


% stress tensor
tau11 = zeros(nx-1,ny-1,nz-1);
tau22 = zeros(nx-1,ny-1,nz-1);
tau33 = zeros(nx-1,ny-1,nz-1);
tau13 = zeros(nx-1,ny-1,nz-1);
tau12 = zeros(nx-1,ny-1,nz-1);
tau23 = zeros(nx-1,ny-1,nz-1);

for kk=1:nz-1
    tau11(:,:,kk) =  2.3438 * 2.0/3.0*k(kk);
    tau22(:,:,kk) =  1.5000 * 2.0/3.0*k(kk);
    tau33(:,:,kk) =  0.6338 * 2.0/3.0*k(kk);
    
    tau13(:,:,kk) = -nu(kk)*dudz(kk);
end

CoEps = zeros(nx-1,ny-1,nz-1);
tke = zeros(nx-1,ny-1,nz-1);

for kk=1:nz-1
    tke(:,:,kk)=k(kk);
    %tke(:,:,k)=( z_cc(k)*C0*uStar^3/0.4/z_cc(k) )^(2/3);
    CoEps(:,:,kk)=C0*eps(kk);
    %CoEps(:,:,k)=C0*uStar^3/0.4/z_cc(k);
end



% now save the netcdf turb output
writeNetCDFFile_turb(filename,x_cc,y_cc,z_cc,CoEps,tke,tau11,tau12,tau13,tau22,tau23,tau33);



