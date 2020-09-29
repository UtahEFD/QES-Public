mainpath='../QES-data';

testcases = {'BaileySinusoidal_4o0_12o0','BaileySinusoidal_0o1_12o0',...
    'BaileySinusoidal_0o05_12o0','BaileySinusoidal_0o01_12o0'};

for k=1:numel(testcases)
    f=testcases{k};
    filename=sprintf('%s/%s_%s.nc',mainpath,f,'particleInfo');
    [particleInfo.(f).data,particleInfo.(f).varname] = readNetCDF(filename);
end

for k=1:numel(testcases)
    f=testcases{k};
    filename=sprintf('%s/%s_%s.nc',mainpath,f,'conc');
    [Concentration.(f).data,Concentration.(f).varname] = readNetCDF(filename);
end

%% ========================================================================
set(0,'defaulttextinterpreter','latex')

figure()
for k=1:numel(testcases)
    f=testcases{k};
    subplot(1,4,k)
    histogram(particleInfo.(f).data.zPos(:,end)/(2*pi),25,...
        'Normalization','pdf','Orientation','horizontal')
    hold all
    plot([1 1],[0 1],'k--')
    ylabel('$z/\delta$')
    xlabel('p.d.f.')
    xlim([0 2.5])
    ylim([0 1])
    
    [ppBin,edgesBin]=histcounts(particleInfo.(f).data.zPos(:,end)/(2*pi),25,'Normalization','pdf');
    S=-sum(ppBin.*log(ppBin))
end

figure()
for k=1:numel(testcases)
    f=testcases{k};
    subplot(1,4,k)
    plot(squeeze(mean(mean(Concentration.(f).data.conc(:,:,:,end),1),2)),Concentration.(f).data.z/(2*pi))
    ylabel('$z/\delta$')
    xlabel('\#part/vol')
end