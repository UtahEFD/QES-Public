% Uniform Flow test case for QES-plume
% Base on Singh PhD Dissertation )
% Initial test case published in 
%  Singh et al. 2004 
%  Willemsen et al. 2007
%
% F. Margaiaraz
% Univesity of Utah. 2021

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

xS=50;yS=20;zS=70;

yProf=[0.393,0.464,0.964]; % streamwise location 

% set the case base name for use in all the other file paths
caseNameWinds = "UniformFlow_yDir";
caseNamePlume = "UniformFlow_yDir_ContRelease";

data=struct();
varnames=struct();

% read wind netcdf file
fileName = sprintf("../QES-data/%s_windsWk.nc",caseNameWinds);
[data.winds,varnames.winds] = readNetCDF(fileName);
% read turb netcdf file
fileName = sprintf("../QES-data/%s_turbOut.nc",caseNameWinds);
[data.turb,varnames.turb] = readNetCDF(fileName);

% read main plume files
fileName = sprintf("../QES-data/%s_plumeOut.nc",caseNamePlume);
[data.plume,varnames.plume] = readNetCDF(fileName);
% read particleInfo files
%fileName = sprintf("../QES-data/%s_particleInfo.nc",caseNamePlume);
%[data.parInfo,varnames.parInfo] = readNetCDF(fileName);

xoH=(data.plume.x)/H;
yoH=(data.plume.y-yS)/H;
zoH=(data.plume.z)/H;

boxDx=mean(diff(data.plume.x));
boxDy=mean(diff(data.plume.y));
boxDz=mean(diff(data.plume.z));
boxVol=double(boxDx*boxDy*boxDz);
CC=dt/tAvg/boxVol;

for k=1:numel(yProf)
    
    idx=find(data.plume.x==xS);
    idy=find(round(1000*yoH)>=round(1000*yProf(k)),1);
    idz=find(data.plume.z==zS);
    
    cStarPlume=squeeze(double(data.plume.pBox(:,idy,:)))*CC*(U*H*H/Q);
    
    sigU=1.78*uStar;
    sigW=(1/(3*0.4*0.4))^(1/3)*uStar;
    
    Ti=(2.5*uStar/zi)^(-1);
    To=1.001; 
    
    y=data.plume.y(idy)-yS;
    t=y/U;
    
    Fx=(1+(t/Ti)^0.5)^(-1);
    %Fz=(1+0.945*(t/To)^0.8)^(-1);
    Fz=Fx;
    
    sigX=sigU*t*Fx;
    sigZ=sigW*t*Fz;
  
    %sigY=0.32*x^0.78;
    %sigZ=0.22*x^0.78;
    
    x=-H:0.1:H;
    z=-H:0.1:H;
    [xx,zz]=ndgrid(x,z);
    
    C=Q/(2*pi*U*sigX*sigZ)*exp(-0.5*xx.^2/sigX^2).*exp(-0.5*zz.^2/sigZ^2);
    cStarModel=C*(U*H*H/Q);
    
    d2plotLat.yoH(k)=y/H;
    d2plotLat.QPlume.xoH=xoH;
    d2plotLat.QPlume.cStar(:,k)=cStarPlume(:,idz);
    d2plotLat.GModel.xoH=x/H+xS/H;
    d2plotLat.GModel.cStar(:,k)=cStarModel(:,floor(numel(z)/2)+1);
    
    d2plotVert.yoH(k)=y/H;
    d2plotVert.QPlume.zoH=zoH;
    d2plotVert.QPlume.cStar(:,k)=cStarPlume(idx,:);
    d2plotVert.GModel.zoH=z/H+zS/H;
    d2plotVert.GModel.cStar(:,k)=cStarModel(floor(numel(x)/2)+1,:);
    
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);
    
    plot(xoH,cStarPlume(:,idz),'s:','LineWidth',2)
    hold all
    plot(x/H+xS/H,cStarModel(:,floor(numel(z)/2)+1),'-','LineWidth',2)
    xlim([.2 1.2])
    
    xlabel('$x/H$')
    ylabel('$C^*$')
    grid on 
    
    %currentPlotName=sprintf('plotOutput/%s_LatConc_%s',...
    %    caseNamePlume,strrep(sprintf('y%.3f',y/H),'.','o'));
    %save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
    
    
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);

    plot(cStarPlume(idx,:),zoH,'s:','LineWidth',2)
    hold all
    plot(cStarModel(floor(numel(x)/2)+1,:),z/H+zS/H,'-','LineWidth',2)
    ylim([.6 1.4])
    
    xlabel('$C^*$')
    ylabel('$z/H$')
    grid on
    
    %currentPlotName=sprintf('plotOutput/%s_VertConc_%s',...
    %    caseNamePlume,strrep(sprintf('y%.3f',y/H),'.','o'));
    %save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
end
%==========================================================================
close all

save('data2plot_yDir','d2plotLat','d2plotVert','caseNameWinds','caseNamePlume')

%==========================================================================

[xx,zz]=ndgrid(data.plume.x-xS,data.plume.z-zS);
yProf2=yoH(yoH > 0.2);
cStarPlume=[];
cStarModel=[];

boxNx=numel(data.plume.x);
boxNy=numel(data.plume.y);
boxNz=numel(data.plume.z);

for k=1:numel(yProf2)
    %================================================
    idy1=find(round(100*yoH)>=round(100*yProf2(k)),1);
    %================================================
    cStarPlume=cat(2,cStarPlume,...
        reshape(squeeze(double(data.plume.pBox(:,idy1,:))*CC*(U*H*H/Q)),[boxNx*boxNz 1]));
    %================================================
    % from Seinfeld and Pandis 1998
    
    sigV=1.78*uStar;
    sigW=(1/(3*0.4*0.4))^(1/3)*uStar;
    
    Ti=(2.5*uStar/zi)^(-1);
    To=1.001; 
    
    y=data.plume.y(idy1)-yS;
    t=y/U;
    
    Fx=(1+(t/Ti)^0.5)^(-1);
    %Fz=(1+0.945*(t/To)^0.8)^(-1);
    Fz=Fx;
    
    sigX=sigV*t*Fx;
    sigZ=sigW*t*Fz;
  
    %sigX=0.32*y^0.78;
    %sigZ=0.22*y^0.78;
    
    
    C=Q/(2*pi*U*sigX*sigZ)*exp(-0.5*xx.^2/sigX^2).*exp(-0.5*zz.^2/sigZ^2);

    cStarModel=cat(2,cStarModel,...
       reshape(C*(U*H*H/Q),[boxNx*boxNz 1]));
    %================================================
    
end

myColorMap=parula(numel(yProf2));
hfig=figure();
hfig.Position=[hfig.Position(1) hfig.Position(2) hfig.Position(3) hfig.Position(3)];
hp=plot(cStarModel,cStarPlume,'kd');
for k=1:numel(yProf2)
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

UniformFlow_yDir_plotPlumeResults
