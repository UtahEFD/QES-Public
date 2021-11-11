% Power Law BL Flow test case for QES-plume
% Base on Singh PhD Dissertation )
% Initial test case published in 
%  Singh et al. 2004 
%  Willemsen et al. 2007
%
% F. Margaiaraz
% Univesity of Utah. 2021
clear all

H=4; % m
p=0.15;
a=4.8;
n=1;
b=0.08;
%b=0.059;

U=a*H^p; % m/s 
uStar=b/0.4; %m/s

alpha=2+p-n;
nu=(1-n)/alpha;

% concentration info
dt=.05; % s
tAvg=1200; % s 
%tAvg=3600; % s 

% source info
Q=20/dt; % #par/s (source strength)
tRelease=1400; % total time of release
Ntot=Q*tRelease; % total number of particles


fsize=12;

xS=20.0;yS=50;zS=4;

%xProf=[6.0,10.0,18.0]; % streamwise location 
xProf=[5.0,10.0,19.0,32.0]; % streamwise location 
%xProf=[5.42,10.97,19.31]; % streamwise location 

% set the case base name for use in all the other file paths
caseNameWinds = "PowerLawBLFlow_long";

%caseNamePlume = "ContRelease_ElevatedReflect";
caseNamePlume = "ContRelease_ElevatedNoReflect";

%caseNamePlume = "ContRelease_xDir";

data=struct();
varnames=struct();

% read wind netcdf file
fileName = sprintf("../QES-data/%s_windsWk.nc",caseNameWinds);
[data.winds,varnames.winds] = readNetCDF(fileName);
% read turb netcdf file
fileName = sprintf("../QES-data/%s_turbOut.nc",caseNameWinds);
[data.turb,varnames.turb] = readNetCDF(fileName);

% read main plume files
fileName = sprintf("../QES-data/%s_conc.nc",caseNamePlume);
[data.plume,varnames.plume] = readNetCDF(fileName);
% read particleInfo files
%fileName = sprintf("../QES-data/%s_particleInfo.nc",caseNamePlume);
%[data.parInfo,varnames.parInfo] = readNetCDF(fileName);

boxNx=numel(data.plume.x);
boxNy=numel(data.plume.y);
boxNz=numel(data.plume.z);

xoH=(data.plume.x-xS)/H;
yoH=(data.plume.y)/H;
zoH=(data.plume.z)/H;

boxDx=mean(diff(data.plume.x));
boxDy=mean(diff(data.plume.y));
boxDz=mean(diff(data.plume.z));
boxVol=double(boxDx*boxDy*boxDz);
CC=dt/tAvg/boxVol;

y=-50:0.01:50;
z=0:0.01:5*H;
[yy,zz]=ndgrid(y,z);

for k=1:numel(xProf)
    %================================================
    idx1=find(round(100*xoH)>=round(100*xProf(k)),1);
    x=data.plume.x(idx1)-xS;
    %================================================
    cStarQES=squeeze(double(data.plume.pBox(idx1,:,:))*CC*(U*H*H/Q));    
    %================================================
    % from Seinfeld and Pandis 1998
    sigY=0.32*x^0.78;
    C=Q/(sqrt(2*pi)*sigY)*exp(-0.5*yy.^2/sigY^2).*...
        exp(-a*(zz.^alpha+H^alpha)/(b*alpha^2*x)).*(zz*H).^(0.5*(1-n))/(b*alpha*x).*...
        besseli(-nu,(2*a*(zz*H).^(0.5*alpha))/(b*alpha^2*x));
    cStarModel=C*(U*H*H/Q);
    %================================================
    idy1=find(data.plume.y>=yS,1);
    idz1=find(data.plume.z>=1.*zS,1)-1;
    %================================================
    idy2=find(y-y(1)>=data.plume.y(idy1),1);
    idz2=find(z>=data.plume.z(idz1),1); 
    %================================================
    
    %================================================
    d2plotLat.xoH(k)=x/H;
    d2plotLat.QPlume.yoH=yoH;
    d2plotLat.QPlume.cStar(:,k)=cStarQES(:,idz1);
    d2plotLat.GModel.yoH=y/H+yS/H;
    d2plotLat.GModel.cStar(:,k)=cStarModel(:,idz2);
    %================================================
    d2plotVert.xoH(k)=x/H;
    d2plotVert.QPlume.zoH=zoH;
    d2plotVert.QPlume.cStar(:,k)=cStarQES(idy1,:);
    d2plotVert.GModel.zoH=z/H;
    d2plotVert.GModel.cStar(:,k)=cStarModel(idy2,:);
    %================================================
    
    %================================================
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);
    
    plot(yoH,cStarQES(:,idz1),'s:','LineWidth',2)
    hold all
    plot((y+yS)/H,cStarModel(:,idz2),'-','LineWidth',2)
    
    xlabel('$y/H$')
    ylabel('$C^*$')
    grid on
    
    currentPlotName=sprintf('plotOutput/%s_%s_LatConc_%s',...
        caseNameWinds,caseNamePlume,strrep(sprintf('x%.2f',x/H),'.','o'));
    save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
    %================================================
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);

    plot(cStarQES(idy1,:),zoH,'s:','LineWidth',2)
    hold all
    plot(cStarModel(idy2,:),z/H,'-','LineWidth',2)
    
    xlabel('$C^*$')
    ylabel('$z/H$')
    grid on 
    
    currentPlotName=sprintf('plotOutput/%s_%s_VertConc_%s',...
           caseNameWinds,caseNamePlume,strrep(sprintf('x%.2f',x/H),'.','o'));
    save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
    %================================================
end
close all
save('data2plot_xDir','d2plotLat','d2plotVert','caseNameWinds','caseNamePlume')

%% ========================================================================
[yy,zz]=ndgrid(data.plume.y-yS,data.plume.z);
xProf2=xoH(xoH > 4 & xoH < 30);
cStarQES=[];
cStarModel=[];
for k=1:numel(xProf2)
    %================================================
    idx1=find(round(100*xoH)>=round(100*xProf2(k)),1);
    x=data.plume.x(idx1)-xS;
    %================================================
    cStarQES=cat(2,cStarQES,...
        reshape(squeeze(double(data.plume.pBox(idx1,:,:))*CC*(U*H*H/Q)),[boxNy*boxNz 1]));
    %================================================
    % from Seinfeld and Pandis 1998
    sigY=0.32*x^0.78;
    C=Q/(sqrt(2*pi)*sigY)*exp(-0.5*yy.^2/sigY^2).*...
        exp(-a*(zz.^alpha+H^alpha)/(b*alpha^2*x)).*(zz*H).^(0.5*(1-n))/(b*alpha*x).*...
        besseli(-nu,(2*a*(zz*H).^(0.5*alpha))/(b*alpha^2*x));
    cStarModel=cat(2,cStarModel,...
       reshape(C*(U*H*H/Q),[boxNy*boxNz 1]));
    %================================================
    
end

myColorMap=parula(numel(xProf2));
hfig=figure();
hfig.Position=[hfig.Position(1) hfig.Position(2) hfig.Position(3) hfig.Position(3)];
hp=plot(cStarQES,cStarModel,'kd');
for k=1:numel(xProf2)
    hp(k).Color=myColorMap(k,:);
    hp(k).MarkerFaceColor=myColorMap(k,:);
end
hold all
grid on
plot([0 2],[0 2],'k--')
xlim([0,1])
xlabel('C^* QES-Plume')
ylim([0,1])
ylabel('C^* non-Gaussian plume model')

SSres=0;
SStot=0;
cStarQESMean=mean(mean(cStarQES));
for k=1:numel(xProf2)
    Ind=(~isnan(cStarModel(:,k)));
    SSres=SSres+sum((cStarQES(Ind,k)-cStarModel(Ind,k)).^2);
    SStot=SStot+sum((cStarQES(Ind,k)-cStarQESMean).^2);
end

R = 1-SSres/SStot;
htxt=text(1,1,sprintf('R^2=%f\n',double(R)));
htxt.Units='normalized';
htxt.Position=[.1 .9 0];

currentPlotName=sprintf('plotOutput/%s_%s_1to1',caseNameWinds,caseNamePlume);
hfig.Units='centimeters';
save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
%% ========================================================================
PowerLawBLFlow_xDir_plotPlumeResults
