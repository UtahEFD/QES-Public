% Uniform Flow test case for QES-plume
% Base on Singh PhD Dissertation )
% Initial test case published in 
%  Singh et al. 2004 
%  Willemsen et al. 2007
%
% F. Margaiaraz
% Univesity of Utah. 2021
clear all

H=70; % m
U=2; % m/s 
uStar=0.174; %m/s
zi=10000; % m

% source info
Q=200; % #par/s (source strength)
tRelease=1400; % total time of release
Ntot=Q*tRelease; % total number of particles

% concentration info
dt=1; % s
tAvg=1200; % s 

fsize=12;

xS=20;yS=50;zS=70;

xProf=[0.393,0.464,0.964]; % streamwise location 

% set the case base name for use in all the other file paths
caseNameWinds = "UniformFlow_xDir";
caseNamePlume = "UniformFlow_xDir_ContRelease";

% set the plotOutputFolders
plotOutputDir = "plotOutput";
mkdir(plotOutputDir)

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

xoH=(data.plume.x-xS)/H;
yoH=(data.plume.y)/H;
zoH=(data.plume.z)/H;

boxDx=mean(diff(data.plume.x));
boxDy=mean(diff(data.plume.y));
boxDz=mean(diff(data.plume.z));
boxVol=double(boxDx*boxDy*boxDz);
CC=dt/tAvg/boxVol;

for k=1:numel(xProf)
    
    idx=find(round(1000*xoH)>=round(1000*xProf(k)),1);
    idy=find(data.plume.y==yS);
    idz=find(data.plume.z==zS);
    
    cStarPlume=squeeze(double(data.plume.pBox(idx,:,:)))*CC*(U*H*H/Q);
    
    sigV=1.78*uStar;
    sigW=(1/(3*0.4*0.4))^(1/3)*uStar;
    
    Ti=(2.5*uStar/zi)^(-1);
    To=1.001; 
    
    x=data.plume.x(idx)-xS;
    t=x/U;
    
    Fy=(1+(t/Ti)^0.5)^(-1);
    %Fz=(1+0.945*(t/To)^0.8)^(-1);
    Fz=Fy;
    
    sigY=sigV*t*Fy;
    sigZ=sigW*t*Fz;
  
    %sigY=0.32*x^0.78;
    %sigZ=0.22*x^0.78;
    
    y=-H:0.1:H;
    z=-H:0.1:H;
    [yy,zz]=ndgrid(y,z);
    
    C=Q/(2*pi*U*sigY*sigZ)*exp(-0.5*yy.^2/sigY^2).*exp(-0.5*zz.^2/sigZ^2);
    cStarModel=C*(U*H*H/Q);
    
    d2plotLat.xoH(k)=x/H;
    d2plotLat.QPlume.yoH=yoH;
    d2plotLat.QPlume.cStar(:,k)=cStarPlume(:,idz);
    d2plotLat.GModel.yoH=y/H+yS/H;
    d2plotLat.GModel.cStar(:,k)=cStarModel(:,floor(numel(z)/2)+1);
    
    d2plotVert.xoH(k)=x/H;
    d2plotVert.QPlume.zoH=zoH;
    d2plotVert.QPlume.cStar(:,k)=cStarPlume(idy,:);
    d2plotVert.GModel.zoH=z/H+zS/H;
    d2plotVert.GModel.cStar(:,k)=cStarModel(floor(numel(y)/2)+1,:);
    
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);
    
    plot(yoH,cStarPlume(:,idz),'s:','LineWidth',2)
    hold all
    plot(y/H+yS/H,cStarModel(:,floor(numel(z)/2)+1),'-','LineWidth',2)
    xlim([.2 1.2])
    
    xlabel('$y/H$')
    ylabel('$C^*$')
    grid on 
    
    %currentPlotName=sprintf('plotOutput/%s_LatConc_%s',...
    %    caseNamePlume,strrep(sprintf('x%.3f',x/H),'.','o'));
    %save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
    
    
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);

    plot(cStarPlume(idy,:),zoH,'s:','LineWidth',2)
    hold all
    plot(cStarModel(floor(numel(y)/2)+1,:),z/H+zS/H,'-','LineWidth',2)
    ylim([.6 1.4])
    
    xlabel('$C^*$')
    ylabel('$z/H$')
    grid on
    
    %currentPlotName=sprintf('plotOutput/%s_VertConc_%s',...
    %    caseNamePlume,strrep(sprintf('x%.3f',x/H),'.','o'));
    %save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
end
%==========================================================================

close all

save('data2plot_xDir','d2plotLat','d2plotVert','caseNameWinds','caseNamePlume')

%==========================================================================

[yy,zz]=ndgrid(data.plume.y-yS,data.plume.z-zS);
xProf2=xoH(xoH > 0.2);
cStarPlume=[];
cStarModel=[];

boxNx=numel(data.plume.x);
boxNy=numel(data.plume.y);
boxNz=numel(data.plume.z);

for k=1:numel(xProf2)
    %================================================
    idx1=find(round(100*xoH)>=round(100*xProf2(k)),1);
    %================================================
    cStarPlume=cat(2,cStarPlume,...
        reshape(squeeze(double(data.plume.pBox(idx1,:,:))*CC*(U*H*H/Q)),[boxNy*boxNz 1]));
    %================================================
    % from Seinfeld and Pandis 1998
    
    sigV=1.78*uStar;
    sigW=(1/(3*0.4*0.4))^(1/3)*uStar;
    
    Ti=(2.5*uStar/zi)^(-1);
    To=1.001; 
    
    x=data.plume.x(idx1)-xS;
    t=x/U;
    
    Fy=(1+(t/Ti)^0.5)^(-1);
    %Fz=(1+0.945*(t/To)^0.8)^(-1);
    Fz=Fy;
    
    sigY=sigV*t*Fy;
    sigZ=sigW*t*Fz;
  
    %sigY=0.32*x^0.78;
    %sigZ=0.22*x^0.78;
    
    
    C=Q/(2*pi*U*sigY*sigZ)*exp(-0.5*yy.^2/sigY^2).*exp(-0.5*zz.^2/sigZ^2);

    cStarModel=cat(2,cStarModel,...
       reshape(C*(U*H*H/Q),[boxNy*boxNz 1]));
    %================================================
    
end

myColorMap=parula(numel(xProf2));
hfig=figure();
hfig.Position=[hfig.Position(1) hfig.Position(2) hfig.Position(3) hfig.Position(3)];
hp=plot(cStarModel,cStarPlume,'kd');
for k=1:numel(xProf2)
    hp(k).Color=myColorMap(k,:);
    hp(k).MarkerFaceColor=myColorMap(k,:);
end
hold all
grid on
plot([0 100],[0 100],'k--')
xlim([0,70])
xlabel('C^* Gaussian plume model')
ylim([0,70])
ylabel('C^* QES-Plume')

SSres=0;
SStot=0;
cStarPlumeMean=mean(mean(cStarPlume));
for k=1:numel(xProf2)
    SSres=SSres+sum((cStarPlume(:,k)-cStarModel(:,k)).^2);
    SStot=SStot+sum((cStarPlume(:,k)-cStarPlumeMean).^2);
end
R = 1-SSres/SStot;

htxt=text(1,1,sprintf('R^2=%f\n',double(R)));
htxt.Units='normalized';
htxt.Position=[.1 .9 0];

currentPlotName=sprintf('plotOutput/%s_1to1',caseNamePlume);
hfig.Units='centimeters';
save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)

%==========================================================================

UniformFlow_xDir_plotPlumeResults
