% dimensions of the 3D domain
lx=2*pi*1000;ly=2*pi*1000;lz=1000;

mainpath='../QES-data';

testcases = {'BaileyLES_22o2_222o0','BaileyLES_2o22_222o0','BaileyLES_0o222_222o0'};
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
    subplot(1,3,k)
    histogram(reshape(particleInfo.(f).data.zPos,[1,numel(particleInfo.(f).data.zPos)])/lz,25,...
        'Normalization','pdf','Orientation','horizontal')
    hold all
    plot([1 1],[0 1],'k--')
    ylabel('$z/\delta$')
    xlabel('p.d.f.')
    xlim([0 1.5])
    ylim([0 1])
end

figure()
for k=1:numel(testcases)
    f=testcases{k};
    subplot(1,3,k)
    plot(squeeze(mean(mean(Concentration.(f).data.conc(:,:,:,end),1),2)),Concentration.(f).data.z/lz)
    ylabel('$z/\delta$')
    xlabel('\#part/vol')
end