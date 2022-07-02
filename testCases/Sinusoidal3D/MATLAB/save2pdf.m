function save2pdf(fighandle,figname,figsize,fontsize)
% =========================================================================
% save figure as pdf
% 
% CALL:  mySave2pdf(hfig,fname,figsize,fontsize)
% INPUT: fighandle - figure hanlde
%        figsize - 2D array [x,y] size of the figure in cm
%        figname - file name
%        fontsize - font size (optional)
%
% Fabien Margairaz, University of Utah, SLC
% =========================================================================
if(exist('fsize'))
    haxes = findobj(fighandle, 'type', 'axes');
    set(haxes,'Fontsize',fontsize)
end

set(fighandle,'PaperUnits','centimeters');
set(fighandle,'PaperSize',figsize);
set(fighandle,'PaperPosition', [0 0 figsize(1) figsize(2)]);

figname=sprintf('%s.pdf',figname);
print(fighandle,'-dpdf',figname);

end
