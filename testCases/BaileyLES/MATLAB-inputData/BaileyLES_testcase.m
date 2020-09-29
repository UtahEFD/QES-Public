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
writeNetCDFFile_winds('BaileyLES',nx,ny,nz,x_cc,y_cc,z_cc,u_out,v_out,w_out,icellflag);


%% ========================================================================
% QES-TURB data:

% tau11
periodicVar=makePeriodic(txx,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
tau11=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
tau11=permute(tau11,[2,1,3]);
% lower BC
tau11(:,:,1)=tau11(:,:,2);
% upper BC
tau11(:,:,33)=tau11(:,:,32);

% tau11
periodicVar=makePeriodic(tyy,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
tau22=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
tau22=permute(tau22,[2,1,3]);
% lower BC
tau22(:,:,1)=tau22(:,:,2);
% upper BC
tau22(:,:,33)=tau22(:,:,32);

% tau11
periodicVar=makePeriodic(tzz,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
tau33=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
tau33=permute(tau33,[2,1,3]);
% lower BC
tau33(:,:,1)=tau33(:,:,2);
% upper BC
tau33(:,:,33)=tau33(:,:,32);

% tau11
periodicVar=makePeriodic(txy,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
tau12=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
tau12=permute(tau12,[2,1,3]);
% lower BC
tau12(:,:,1)=tau12(:,:,2);
% upper BC
tau12(:,:,33)=tau12(:,:,32);

% tau11
periodicVar=makePeriodic(txz,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
tau13=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
tau13=permute(tau13,[2,1,3]);
% lower BC
tau13(:,:,1)=tau13(:,:,2);
% upper BC
tau13(:,:,33)=tau13(:,:,32);

% tau11
periodicVar=makePeriodic(tyz,nx,ny,nz,2);
[xxq,yyq,zzq]=meshgrid(x_cc,y_cc,z_cc);
permVar=permute(periodicVar,[2,1,3]);
tau23=interp3(xx,yy,zz,permVar,xxq,yyq,zzq);
tau23=permute(tau23,[2,1,3]);
% lower BC
tau23(:,:,1)=tau23(:,:,2);
% upper BC
tau23(:,:,33)=tau23(:,:,32);

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
writeNetCDFFile_turb('BaileyLES',x_cc,y_cc,z_cc,CoEps,tke,tau11,tau12,tau13,tau22,tau23,tau33);


%% ========================================================================
% some figure to vizualize the data generated vs original data:
set(0,'defaulttextinterpreter','latex')

% U velocity 
figure()
plot(squeeze(mean(mean(u,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(u_out(:,1:end-1,1:end-1),1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$u/u_*$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% V velocity 
figure()
plot(squeeze(mean(mean(v,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(v_out(1:end-1,:,1:end-1),1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$v/u_*$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% W velocity 
figure()
plot(squeeze(mean(mean(w,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(w_out(1:end-1,1:end-1,:),1),2)),z_pp/lz,'o')
ylabel('$z/\delta$')
xlabel('$w/u_*$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

%% ========================================================================
% some figure to vizualize the data generated vs original data:
set(0,'defaulttextinterpreter','latex')

% tau11 stress 
figure()
plot(squeeze(mean(mean(txx,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau11,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{xx}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% tau22 stress 
figure()
plot(squeeze(mean(mean(tyy,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau22,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{yy}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% tau33 stress 
figure()
plot(squeeze(mean(mean(tzz,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau33,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{zz}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')


% tau12 stress 
figure()
plot(squeeze(mean(mean(txy,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau12,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{xy}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% tau13 stress 
figure()
plot(squeeze(mean(mean(txz,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau13,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{xz}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% tau23 stress 
figure()
plot(squeeze(mean(mean(tyz,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau23,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{yz}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% epps dissipation rate 
figure()
plot(squeeze(mean(mean(epps,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(epps_out,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\epsilon$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')




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