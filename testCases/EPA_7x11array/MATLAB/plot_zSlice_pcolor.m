% Uniform Flow test case for QES-plume
% Base on Singh PhD Dissertation )
% Initial test case published in 
%  Singh et al. 2004 
%  Willemsen et al. 2007
%
% F. Margaiaraz
% Univesity of Utah. 2021
%==========================================================================
figW=23;figH=23;fsize=14;
%========================
% nProf=numel(d2plotLat.xoH);
% cStr=char(96+(1:2*nProf));
%========================
hfig=figure;
hfig.Units='centimeters';hfig.Position=[0 -20 figW figH];
set(hfig,'Units','centimeters')
set(hfig,'defaulttextinterpreter','latex','DefaultAxesFontSize',fsize)
[haxes,axpos]=tightSubplot(1,1,[2.0/figH 1.5/figW],1/figH*[1.5 0.5],1/figW*[2 0.5]);

pcolor(xoH,yoH,CC*(U*H*H/Q)*double(squeeze(data.plume.pBox(:,:,idz))'))

xlabel('$x/H$')
xlim([-2 22])

xlabel('$y/H$')
ylim([-12 12])

set(haxes,'ColorScale','log')
shading flat
caxis([.001 1])
colormap(flipud(bone))
