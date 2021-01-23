% Power Law BL Flow test case for QES-plume
% Base on Singh PhD Dissertation )
% Initial test case published in 
%  Singh et al. 2004 
%  Willemsen et al. 2007
%
% F. Margaiaraz
% Univesity of Utah. 2021

H=4; % m
U=5.90; % m/s 
uStar=0.2; %m/s

% source info
Q=100; % #par/s (source strength)
tRelease=1000; % total time of release
Ntot=Q*tRelease; % total number of particles

% concentration info
dt=1; % s
tAvg=800; % s 

fsize=12;

xS=20;yS=50;zS=4;

xProf=[10.97,19.31]; % streamwise location 

% set the case base name for use in all the other file paths
caseNameWinds = "PowerLawBLFlow";
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


p=0.15;
a=4.8;
n=1;
b=0.059;

alpha=2+p-n;
nu=(1-n)/alpha;

y=-50:0.1:50;
z=0:0.01:5*H;
[yy,zz]=ndgrid(y,z);

for k=1:numel(xProf)
    
    idx1=find(round(100*xoH)>=round(100*xProf(k)),1);
    idy1=find(data.plume.y>=yS,1);
    idz1=find(data.plume.z>=zS,1);
    
    cStarPlume=squeeze(double(data.plume.pBox(idx1,:,:))*CC*(U*H*H/Q));
    
    x=data.plume.x(idx1)-xS;
    
    idy2=find(y-y(1)>=data.plume.y(idy1),1);
    idz2=find(z>=data.plume.z(idz1),1); 
    
    % from Seinfeld and Pandis 1998
    sigY=0.32*x^0.78;
    
    C=Q/((2*pi)^0.5*sigY)*exp(-0.5*yy.^2/sigY^2).*...
        exp(-a*(zz.^alpha+H^alpha)/(b*alpha^2*x)).*(zz*H).^(0.5*(1-n))/(b*alpha*x).*...
        besseli(-nu,(2*a*(zz*H).^(0.5*alpha))/(b*alpha^2*x));
    cStarModel=C*(U*H*H/Q);
    
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);
    
    plot(yoH,cStarPlume(:,idz1),'s:','LineWidth',2)
    hold all
    plot((y+yS)/H,cStarModel(:,idz2),'-','LineWidth',2)
    %xlim([.2 1.2])
    
    xlabel('$y/H$')
    ylabel('$C^*$')
    
    currentPlotName=sprintf('plotOutput/%s_%s_LatConc_%s',...
        caseNameWinds,caseNamePlume,strrep(sprintf('x%.2f',x/H),'.','o'));
    save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
    
    hfig = figure;
    set(hfig,'Units','centimeters')
    set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
    [haxes,axpos]=tightSubplot(1,1,[.02 .02],[.15 .05],[.15 .05]);

    plot(cStarPlume(idy1,:),zoH,'s:','LineWidth',2)
    hold all
    plot(cStarModel(idy2,:),z/H,'-','LineWidth',2)

    %ylim([.6 1.4])
    
    xlabel('$C^*$')
    ylabel('$z/H$')
    
    currentPlotName=sprintf('plotOutput/%s_%s_VertConc_%s',...
           caseNameWinds,caseNamePlume,strrep(sprintf('x%.2f',x/H),'.','o'));
    save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)
end

