% channel 3D case
lx=1;ly=1;lz=1;
nx=51;ny=51;nz=52;

dx=lx/(nx-1);
dy=ly/(ny-1);
dz=lz/(nz-2);

x_cc=0.5*dx:dx:lx-0.5*dx;
y_cc=0.5*dy:dy:ly-0.5*dy;
z_cc=-0.5*dz:dz:lz-0.5*dz;



uMean_file = 'channel_u.txt';
C0 = 4.0;
epps_file = 'channel_epps.txt';
sigma2_file = 'channel_sigma2.txt';

uMean = load(uMean_file);
zchannel = linspace(0,lz,length(uMean));
epps = load(epps_file);
sigma2 = load(sigma2_file);

% QES-WINDS data:
u_out = zeros(nx,ny,nz);

u_new=interp1(zchannel,uMean,z_cc);
u_new(1)=0;

for k=1:nz-1
    u_out(:,:,k)=u_new(k);
end
    
icellflag_out = ones(nx-1,ny-1,nz-1);
icellflag_out(:,:,1) = 2; % terrain

% data for NetCDF file
u = u_out; 
v = zeros(nx,ny,nz);
w = zeros(nx,ny,nz);

icellflag = icellflag_out;


% QES-TURB data:

sig2_new=interp1(zchannel,sigma2,z_cc);
sig2_new(1)=0; % ghost cell
epps_new=interp1(zchannel,epps,z_cc);
epps_new(1)=0; % ghost cell 
epps_new(2)=epps(2); % 1st point (to get corret wall dissipation)

sig2_out = zeros(nx-1,ny-1,nz-1);
tke_out = zeros(nx-1,ny-1,nz-1);
CoEps_out = zeros(nx-1,ny-1,nz-1);

for k=1:nz-1
    sig2_out(:,:,k)=sig2_new(k);
    CoEps_out(:,:,k)=C0*epps_new(k);
    tke_out(:,:,k)=(z_cc(k)*epps_new(k))^(2/3);
end

% data for NetCDF file
tau11 = sig2_out;
tau12 = zeros(nx-1,ny-1,nz-1);
tau13 = zeros(nx-1,ny-1,nz-1);
tau22 = sig2_out;
tau23 = zeros(nx-1,ny-1,nz-1);
tau33 = sig2_out;
tke = tke_out;
CoEps = CoEps_out;


figure()
plot(uMean,zchannel,'x')
hold all
plot(u_new,z_cc,'o')
ylabel('$z/\delta$')
xlabel('$u/u_*$')


figure()
plot(sigma2,zchannel,'x')
hold all
plot(sig2_new,z_cc,'o')
ylabel('$z/\delta$')
xlabel('$\sigma^2/u_*^2$')

figure()
plot(epps,zchannel,'x')
hold all
plot(epps_new,z_cc,'o')
ylabel('$z/\delta$')
xlabel('$\varepsilon\delta/u_*^3$')

figure()
plot((z_cc.*epps_new).^(2/3),z_cc,'o')
ylabel('$z/\delta$')
xlabel('$tke/u_*^2$')

% now save the netcdf wind output
writeNetCDFFile_winds('channel3D',nx,ny,nz,x_cc,y_cc,z_cc,u,v,w,icellflag);

% now save the netcdf turb output
writeNetCDFFile_turb('channel3D',x_cc,y_cc,z_cc,CoEps,tke,tau11,tau12,tau13,tau22,tau23,tau33);