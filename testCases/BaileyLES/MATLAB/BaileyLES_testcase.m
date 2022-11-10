% Bailes LES test case for QES-plume
% Base on Bailey 2017 (BLM)
%
% F. Margaiaraz
% Univesity of Utah. 2020

%% ========================================================================
% setup:

% dimensions of the 3D domain
lx=2*pi*1000;ly=2*pi*1000;lz=1000;

% grid resolution in x,y,z is set to have:
% 32 faces in the plume domain -> 2 extra face on top (before) and bottom (after)
% -> 31 cell within the plume domain -> 2 extra cell on top (before) and bottom (after)
nx=34;ny=34;nz=34;

% 32 faces 31 cell within the plume domain
dx=lx/(nx-3);dy=ly/(ny-3);
% 32 faces 31 cell within the plume domain
dz=lz/(nz-3); 

% grid definition for cell center variables
x_cc=-0.5*dx:dx:lx+0.5*dx;
y_cc=-0.5*dy:dy:ly+0.5*dy;
% grid definition for cell center variables -> 2 extra cell on top (above) and bottom (below)
z_cc=-0.5*dz:dz:lz+0.5*dz;

% loading original data set
x=linspace(0,lx,32);
y=linspace(0,ly,32);
z=linspace(0,lz,32);

x_pp=[-dx x lx+dx];
y_pp=[-dy y ly+dy];
z_pp=[-dz z lz+dz];
[xx,yy,zz]=meshgrid(x_pp,y_pp,z_pp);

C0 = 4.0;
u=reshape(load("LES_u.txt"),[32 32 32]);
v=reshape(load("LES_v.txt"),[32 32 32]);
w=reshape(load("LES_w.txt"),[32 32 32]);

txx=reshape(load("LES_txx.txt"),[32 32 32]);
tyy=reshape(load("LES_tyy.txt"),[32 32 32]);
tzz=reshape(load("LES_tzz.txt"),[32 32 32]);
txy=reshape(load("LES_txy.txt"),[32 32 32]);
txz=reshape(load("LES_txz.txt"),[32 32 32]);
tyz=reshape(load("LES_tyz.txt"),[32 32 32]);

sigma2=reshape(load("LES_sigma2.txt"),[32 32 32]);
epps=reshape(load("LES_epps.txt"),[32 32 32]);

%% ========================================================================
% QES-WINDS data:

% u-vlocity
periodicVar=makePeriodic(u,nx,ny,nz,1);
[xxq,yyq,zzq]=meshgrid(x_pp,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
u_out=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
u_out=permute(u_out,[2,1,3]);
% add extra layers for QES-Winds data shape
u_out(:,34,:)=0;u_out(:,:,34)=0;
% lower BC
u_out(:,:,1)=-u_out(:,:,2);
% upper BC
u_out(:,:,33)=u_out(:,:,32);

% v-vlocity
periodicVar=makePeriodic(v,nx,ny,nz,1);
[xxq,yyq,zzq]=meshgrid(x_cc,y_pp,z_cc);
permVar=permute(periodicVar,[2,1,3]);
v_out=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
v_out=permute(v_out,[2,1,3]);
% add extra layers for QES-Winds data shape
v_out(34,:,:)=0;v_out(:,:,34)=0;
% lower BC
v_out(:,:,1)=-v_out(:,:,2);
% upper BC
v_out(:,:,33)=v_out(:,:,32);

% w-vlocity
periodicVar=makePeriodic(w,nx,ny,nz,1);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_pp);
permVar=permute(periodicVar,[2,1,3]);
w_out=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
w_out=permute(w_out,[2,1,3]);
% add extra layers for QES-Winds data shape
w_out(34,:,:)=0;w_out(:,34,:)=0;
% lower ghost cell
w_out(:,:,1)=0;
% upper ghost cell
w_out(:,:,34)=0;

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
writeNetCDFFile_winds('../QES-data/BaileyLES',nx,ny,nz,x_cc,y_cc,z_cc,u_out,v_out,w_out,icellflag);


%% ========================================================================
% QES-TURB data:

% txx
periodicVar=makePeriodic(txx,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
txx=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
txx=permute(txx,[2,1,3]);
% lower BC
txx(:,:,1)=txx(:,:,2);
% upper BC
txx(:,:,33)=txx(:,:,32);

% txx
periodicVar=makePeriodic(tyy,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
tyy=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
tyy=permute(tyy,[2,1,3]);
% lower BC
tyy(:,:,1)=tyy(:,:,2);
% upper BC
tyy(:,:,33)=tyy(:,:,32);

% txx
periodicVar=makePeriodic(tzz,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
tzz=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
tzz=permute(tzz,[2,1,3]);
% lower BC
tzz(:,:,1)=tzz(:,:,2);
% upper BC
tzz(:,:,33)=tzz(:,:,32);

% txx
periodicVar=makePeriodic(txy,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
txy=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
txy=permute(txy,[2,1,3]);
% lower BC
txy(:,:,1)=txy(:,:,2);
% upper BC
txy(:,:,33)=txy(:,:,32);

% txx
periodicVar=makePeriodic(txz,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
txz=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
txz=permute(txz,[2,1,3]);
% lower BC
txz(:,:,1)=txz(:,:,2);
% upper BC
txz(:,:,33)=txz(:,:,32);

% txx
periodicVar=makePeriodic(tyz,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
tyz=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
tyz=permute(tyz,[2,1,3]);
% lower BC
tyz(:,:,1)=tyz(:,:,2);
% upper BC
tyz(:,:,33)=tyz(:,:,32);

% epps
periodicVar=makePeriodic(epps,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
epps_out=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
epps_out=permute(epps_out,[2,1,3]);
% lower BC
epps_out(:,:,1)=epps_out(:,:,2);
% upper BC
epps_out(:,:,33)=epps_out(:,:,32);

tke=zeros(nx-1,ny-1,nz-1);
for k=2:nz-1
    tke(:,:,k)=(z_cc(k)*epps_out(k))^(2/3);
end

CoEps=C0*epps_out;

% now save the netcdf turb output
writeNetCDFFile_turb('../QES-data/BaileyLES',x_cc,y_cc,z_cc,CoEps,tke,txx,txy,txz,tyy,tyz,tzz);

%% ========================================================================
function out = makePeriodic(in,nx,ny,nz,lbc)
out=zeros(nx,ny,nz);
out(2:end-1,2:end-1,2:end-1)=in;

% x-values
out(1,2:end-1,2:end-1)=in(end,:,:);
out(end,2:end-1,2:end-1)=in(1,:,:);
% y-values
out(2:end-1,1,2:end-1)=in(:,end,:);
out(2:end-1,end,2:end-1)=in(:,1,:);

% corners values
out(1,1,:)=out(end-1,end-1,:);
out(end,end,:)=out(2,2,:);
out(end,1,:)=out(2,end-1,:);
out(1,end,:)=out(end-1,2,:);

% boundary conditions (bottom)
if(lbc==1)
    out(:,:,1)=-out(:,:,3);
elseif(lbc==2)
    out(:,:,1)=out(:,:,3);
    out(:,:,2)=out(:,:,3);
end

% boundary conditions (top)
out(:,:,end)=out(:,:,end-1);

end
