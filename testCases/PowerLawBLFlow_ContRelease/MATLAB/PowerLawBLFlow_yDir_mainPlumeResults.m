% Power Law BL Flow test case for QES-plume
% Base on Singh PhD Dissertation )
% Initial test case published in 
%  Singh et al. 2004 
%  Willemsen et al. 2007
%
% F. Margaiaraz
% Univesity of Utah. 2021

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

% source info
Q=200; % #par/s (source strength)
tRelease=2000; % total time of release
Ntot=Q*tRelease; % total number of particles

% concentration info
dt=1; % s
tAvg=1800; % s 

fsize=12;

% set the plotOutputFolders
plotOutputDir = "plotOutput";
mkdir(plotOutputDir)

xS=50;yS=20;zS=4;

yProf=[6.0,10.0,19.0]; % streamwise location 
%xProf=[5.42,10.97,19.31]; % streamwise location 

% set the case base name for use in all the other file paths
caseNameWinds = "PowerLawBLFlow_yDir";
caseNamePlume = "PowerLawBLFlow_yDir_ContRelease";

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

x=-50:0.1:50;
z=0:0.1:5*H;
[xx,zz]=ndgrid(x,z);

for k=1:numel(yProf)
    %================================================
    idy1=find(round(100*yoH)>=round(100*yProf(k)),1);
    y=data.plume.y(idy1)-yS;
    %================================================
    cStarPlume=squeeze(double(data.plume.pBox(:,idy1,:))*CC*(U*H*H/Q));    
    %================================================
    % from Seinfeld and Pandis 1998
    sigX=0.32*y^0.78;
    C=Q/(sqrt(2*pi)*sigX)*exp(-0.5*xx.^2/sigX^2).*...
        exp(-a*(zz.^alpha+H^alpha)/(b*alpha^2*y)).*(zz*H).^(0.5*(1-n))/(b*alpha*y).*...
        besseli(-nu,(2*a*(zz*H).^(0.5*alpha))/(b*alpha^2*y));
    cStarModel=C*(U*H*H/Q);
    %================================================
    idx1=find(data.plume.x>=xS,1);
    idz1=find(data.plume.z>=zS,1);
    %================================================
    idx2=find(x-x(1)>=data.plume.x(idx1),1);
    idz2=find(z>=data.plume.z(idz1),1); 
    %================================================
    
    %================================================
    d2plotLat.yoH(k)=y/H;
    d2plotLat.QPlume.xoH=xoH;
    d2plotLat.QPlume.cStar(:,k)=cStarPlume(:,idz1);
    d2plotLat.GModel.xoH=x/H+xS/H;
    d2plotLat.GModel.cStar(:,k)=cStarModel(:,idz2);
    %================================================
    d2plotVert.yoH(k)=y/H;
    d2plotVert.QPlume.zoH=zoH;
    d2plotVert.QPlume.cStar(:,k)=cStarPlume(idx1,:);
    d2plotVert.GModel.zoH=z/H;
    d2plotVert.GModel.cStar(:,k)=cStarModel(idx2,:);
    %================================================
    
    %================================================
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);
    
    plot(xoH,cStarPlume(:,idz1),'s:','LineWidth',2)
    hold all
    plot((x+xS)/H,cStarModel(:,idz2),'-','LineWidth',2)
    
    xlabel('$x/H$')
    ylabel('$C^*$')
    grid on
    
    %currentPlotName=sprintf('plotOutput/%s_%s_LatConc_%s',...
    %    caseNameWinds,caseNamePlume,strrep(sprintf('y%.2f',y/H),'.','o'));
    %save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
    %================================================
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);

    plot(cStarPlume(idx1,:),zoH,'s:','LineWidth',2)
    hold all
    plot(cStarModel(idx2,:),z/H,'-','LineWidth',2)
    
    xlabel('$C^*$')
    ylabel('$z/H$')
    grid on 
    
    %currentPlotName=sprintf('plotOutput/%s_%s_VertConc_%s',...
    %       caseNameWinds,caseNamePlume,strrep(sprintf('y%.2f',y/H),'.','o'));
    %save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
    %================================================
end
close all

save('data2plot_yDir','d2plotLat','d2plotVert','caseNameWinds','caseNamePlume')
%==========================================================================

[xx,zz]=ndgrid(data.plume.x-xS,data.plume.z);
yProf2=yoH(yoH > 2);
cStarPlume=[];
cStarModel=[];

boxNx=numel(data.plume.x);
boxNy=numel(data.plume.y);
boxNz=numel(data.plume.z);

for k=1:numel(yProf2)
    %================================================
    idy1=find(round(100*yoH)>=round(100*yProf2(k)),1);
    y=data.plume.y(idy1)-yS;
    %================================================
    cStarPlume=cat(2,cStarPlume,...
        reshape(squeeze(double(data.plume.pBox(:,idy1,:))*CC*(U*H*H/Q)),[boxNx*boxNz 1]));
    %================================================
    % from Seinfeld and Pandis 1998
    
    sigX=0.32*y^0.78;
    
    C=Q/(sqrt(2*pi)*sigX)*exp(-0.5*xx.^2/sigX^2).*...
        exp(-a*(zz.^alpha+H^alpha)/(b*alpha^2*y)).*(zz*H).^(0.5*(1-n))/(b*alpha*y).*...
        besseli(-nu,(2*a*(zz*H).^(0.5*alpha))/(b*alpha^2*y));
    
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
plot([0 1],[0 1],'k--')
xlim([0,.6])
xlabel('C^* Gaussian plume model')
ylim([0,.6])
ylabel('C^* QES-Plume')

SSres=0;
SStot=0;
cStarPlumeMean=mean(mean(cStarPlume));
for k=1:numel(yProf2)
    validInd=~isnan(cStarModel(:,k));
    SSres=SSres+sum((cStarPlume(validInd,k)-cStarModel(validInd,k)).^2);
    SStot=SStot+sum((cStarPlume(validInd,k)-cStarPlumeMean).^2);
end
R = 1-SSres/SStot;

htxt=text(1,1,sprintf('R^2=%f\n',double(R)));
htxt.Units='normalized';
htxt.Position=[.1 .9 0];

currentPlotName=sprintf('plotOutput/%s_1to1',caseNamePlume);
hfig.Units='centimeters';
save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)

%==========================================================================
PowerLawBLFlow_yDir_plotPlumeResults

