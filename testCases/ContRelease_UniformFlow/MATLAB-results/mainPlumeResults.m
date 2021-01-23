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
uStar=0.18; %m/s

% source info
Q=100; % #par/s (source strength)
tRelease=1000; % total time of release
Ntot=Q*tRelease; % total number of particles

% concentration info
dt=1; % s
tAvg=800; % s 

fsize=12;

xS=20;yS=50;zS=70;

xProf=[0.393,0.464]; % streamwise location 

% set the case base name for use in all the other file paths
caseNameWinds = "UniformFlow";
caseNamePlume = "ContRelease";

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
fileName = sprintf("../QES-data/%s_particleInfo.nc",caseNamePlume);
[data.parInfo,varnames.parInfo] = readNetCDF(fileName);

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
    sigW=uStar*(1/(3*0.4*0.4))^(1/3);
    
    Ti=2.5*uStar/1000;
    T0=1.001; 
    
    x=data.plume.x(idx)-xS;
    t=x/U;
    
    Fy=1;%(1+(t/Ti)^0.5)^(-1);
    Fz=1;%(1+0.945*(t/T0)^0.8)^(-1);
    
    sigY=sigV*t*Fy;
    sigZ=sigW*t*Fz;
  
    y=-H:0.1:H;
    z=-H:0.1:H;
    [yy,zz]=ndgrid(y,z);
    
    C=Q/(2*pi*U*sigY*sigZ)*exp(-yy.^2/sigY^2).*exp(-zz.^2/sigZ^2);
    cStarModel=C*(U*H*H/Q);
    
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
    
    currentPlotName=sprintf('plotOutput/%s_%s_LatConc_%s',...
        caseNameWinds,caseNamePlume,strrep(sprintf('x%.3f',x/H),'.','o'));
    save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
    
    zH=-1:0.001:1;
    C=1/(2*pi*sigY*sigZ)*exp(-zH.^2/sigZ^2);
    
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
    
    currentPlotName=sprintf('plotOutput/%s_%s_VertConc_%s',...
        caseNameWinds,caseNamePlume,strrep(sprintf('x%.3f',x/H),'.','o'));
    save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
end

