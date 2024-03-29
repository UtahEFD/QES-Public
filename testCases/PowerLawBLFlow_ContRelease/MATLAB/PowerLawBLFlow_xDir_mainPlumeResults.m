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



% concentration info
dt=1.0; % s
%tAvg=1200; % s 
tAvg=1800; % s 

% source info
Q=200/dt; % #par/s (source strength)
tRelease=2100; % total time of release
Ntot=Q*tRelease; % total number of particles

fsize=12;

% set the plotOutputFolders
plotOutputDir = "plotOutput";
mkdir(plotOutputDir)

xS=20;yS=50;zS=4;

xProf=[6.0,10.0,19.0]; % streamwise location 
%xProf=[4.0,10.0,18.0,36.0]; % streamwise location 
%xProf=[5.42,10.97,19.31]; % streamwise location 


% set the case base name for use in all the other file paths
caseNameWinds = "PowerLawBLFlow_xDir";
%caseNameWinds = "PowerLawBLFlow_long";
%caseNamePlume = "ContRelease_ElevatedReflect";
%caseNamePlume = "ContRelease_ElevatedNoReflect";
%caseNamePlume = "ContRelease_NearSurfaceNoReflect";
%caseNamePlume = "ContRelease_NearSurfaceNoReflect";
caseNamePlume = "PowerLawBLFlow_xDir_ContRelease";

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
    cStarPlume=squeeze(double(data.plume.pBox(idx1,:,:))*CC*(U*H*H/Q));    
    %================================================
    % from Seinfeld and Pandis 1998
    
    %t=x/U;
    %sigV=1.78*uStar;
    %Ti=(2.5*uStar/zi)^(-1);
    %Fy=(1+(t/Ti)^0.5)^(-1);
    %sigY=sigV*t*Fy
    
    sigY=0.32*x^0.78;
    %sigY=1.8*0.2*x/U;
    
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
    d2plotLat.QPlume.cStar(:,k)=cStarPlume(:,idz1);
    d2plotLat.GModel.yoH=y/H+yS/H;
    d2plotLat.GModel.cStar(:,k)=cStarModel(:,idz2);
    %================================================
    d2plotVert.xoH(k)=x/H;
    d2plotVert.QPlume.zoH=zoH;
    d2plotVert.QPlume.cStar(:,k)=cStarPlume(idy1,:);
    d2plotVert.GModel.zoH=z/H;
    d2plotVert.GModel.cStar(:,k)=cStarModel(idy2,:);
    %================================================
    
    %================================================
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);
    
    plot(yoH,cStarPlume(:,idz1),'s:','LineWidth',2)
    hold all
    plot((y+yS)/H,cStarModel(:,idz2),'-','LineWidth',2)
    
    xlabel('$y/H$')
    ylabel('$C^*$')
    grid on
    
    %currentPlotName=sprintf('plotOutput/%s_%s_LatConc_%s',...
    %    caseNameWinds,caseNamePlume,strrep(sprintf('x%.2f',x/H),'.','o'));
    %save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
    %================================================
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);

    plot(cStarPlume(idy1,:),zoH,'s:','LineWidth',2)
    hold all
    plot(cStarModel(idy2,:),z/H,'-','LineWidth',2)
    
    xlabel('$C^*$')
    ylabel('$z/H$')
    grid on 
    
    %currentPlotName=sprintf('plotOutput/%s_%s_VertConc_%s',...
    %       caseNameWinds,caseNamePlume,strrep(sprintf('x%.2f',x/H),'.','o'));
    %save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
    %================================================
end
close all

save('data2plot_xDir','d2plotLat','d2plotVert','caseNameWinds','caseNamePlume')
%==========================================================================

[yy,zz]=ndgrid(data.plume.y-yS,data.plume.z);
xProf2=xoH(xoH > 2);
cStarPlume=[];
cStarModel=[];

boxNx=numel(data.plume.x);
boxNy=numel(data.plume.y);
boxNz=numel(data.plume.z);

for k=1:numel(xProf2)
    %================================================
    idx1=find(round(100*xoH)>=round(100*xProf2(k)),1);
    x=data.plume.x(idx1)-xS;
    %================================================
    cStarPlume=cat(2,cStarPlume,...
        reshape(squeeze(double(data.plume.pBox(idx1,:,:))*CC*(U*H*H/Q)),[boxNy*boxNz 1]));
    %================================================
    % from Seinfeld and Pandis 1998
    
    %t=x/U;
    %sigV=1.78*uStar;
    %Ti=(2.5*uStar/zi)^(-1);
    %Fy=(1+(t/Ti)^0.5)^(-1);
    %sigY=sigV*t*Fy;
    
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
hp=plot(cStarModel,cStarPlume,'kd');
for k=1:numel(xProf2)
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
maxRelErr=zeros(size(xProf2));

cStarPlumeMean=mean(mean(cStarPlume));
for k=1:numel(xProf2)
    validInd=~isnan(cStarModel(:,k));
    SSres=SSres+sum((cStarPlume(validInd,k)-cStarModel(validInd,k)).^2);
    SStot=SStot+sum((cStarPlume(validInd,k)-cStarPlumeMean).^2);
    maxRelErr(k) = max(abs(cStarPlume(validInd,k)-cStarModel(validInd,k)))/(max(cStarModel(validInd,k)));
end
R = 1-SSres/SStot;

RMSE = sqrt(SSres/(boxNy*boxNz*numel(xProf2)));

htxt=text(1,1,sprintf('$r^2$=%f\n',double(R)));
htxt.Units='normalized';
htxt.Interpreter='latex';
htxt.FontSize=12;
htxt.Position=[.1 .9 0];

htxt=text(1,1,sprintf('MaxRelErr=%f\n',double(max(maxRelErr))));
htxt.Units='normalized';
htxt.Interpreter='latex';
htxt.FontSize=12;
htxt.Position=[.1 .8 0];

htxt=text(1,1,sprintf('RMSE=%f\n',double(RMSE)));
htxt.Units='normalized';
htxt.Interpreter='latex';
htxt.FontSize=12;
htxt.Position=[.1 .7 0];

currentPlotName=sprintf('plotOutput/%s_1to1',caseNamePlume);
hfig.Units='centimeters';
save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)

%==========================================================================
PowerLawBLFlow_xDir_plotPlumeResults
