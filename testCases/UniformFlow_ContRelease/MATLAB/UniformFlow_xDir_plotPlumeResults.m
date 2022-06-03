% Uniform Flow test case for QES-plume
% Base on Singh PhD Dissertation )
% Initial test case published in 
%  Singh et al. 2004 
%  Willemsen et al. 2007
%
% F. Margaiaraz
% Univesity of Utah. 2021
%==========================================================================
figW=30;figH=20;fsize=14;
%========================
load('data2plot_xDir.mat')
nProf=numel(d2plotLat.xoH);
cStr=char(96+(1:2*nProf));
%========================
hfig=figure;
hfig.Units='centimeters';hfig.Position=[0 -20 figW figH];
set(hfig,'Units','centimeters')
set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
[haxes,axpos]=tightSubplot(2,nProf,[2.0/figH 1.5/figW],1/figH*[1.5 0.5],1/figW*[2 0.5]);

for k=1:nProf
    axes(haxes(k))
    plot(d2plotLat.QPlume.yoH,d2plotLat.QPlume.cStar(:,k),'s','LineWidth',2)
    hold all
    plot(d2plotLat.GModel.yoH,d2plotLat.GModel.cStar(:,k),'-','LineWidth',2)
    
    xlim([-.4 .4])
    %ylim([0 65])
    grid on
    xlabel('$y/H$')
    
    %tmpstr=sprintf('(%s) $x/H=$%.3f',cStr(k),d2plotLat.xoH(k));
    tmpstr=sprintf('(%s)',cStr(k));
    htext=text(0.05,0.92,tmpstr,'Units','normalized');
    set(htext,'interpreter','latex','FontSize',fsize,'BackgroundColor','w');
end
%set(haxes(2:nProf),'YtickLabel',[])
%set(haxes(1:nProf),'Xtick',0.2:0.2:1.2)
axes(haxes(1));ylabel('$C^*$')

for k=1:nProf
    axes(haxes(k+nProf))
    plot(d2plotVert.QPlume.cStar(:,k),d2plotVert.QPlume.zoH,'s','LineWidth',2)
    hold all
    plot(d2plotVert.GModel.cStar(:,k),d2plotVert.GModel.zoH,'-','LineWidth',2)
   
    %xlim([0 65])
    ylim([.6 1.4])
    grid on
    xlabel('$C^*$')
    
    tmpstr=sprintf('(%s) $x/H=$%.3f',cStr(k+nProf),d2plotLat.xoH(k));
    htext=text(0.05,0.92,tmpstr,'Units','normalized');
    set(htext,'interpreter','latex','FontSize',fsize,'BackgroundColor','w');
end
%set(haxes(nProf+2:2*nProf),'YtickLabel',[])
axes(haxes(1+nProf));ylabel('$z/H$')

for k=1:nProf
    m1=haxes(k).YLim(2);
    m2=haxes(k+nProf).XLim(2);
    [m3,I]=max([m1,m2]);
    haxes(k).YLim(2)=m3;
    haxes(k+nProf).XLim(2)=m3;
    if I==1
       haxes(k+nProf).XTick=haxes(k).YTick;
    else
       haxes(k).XTick=haxes(k+nProf).YTick;
    end
end

currentPlotName=sprintf('plotOutput/%s_ModelComp',caseNamePlume);
save2pdf(hfig,currentPlotName,hfig.Position(3:4),12)

