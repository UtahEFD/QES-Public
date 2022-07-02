%% ========================================================================
% some figure to vizualize the data generated vs original data:
set(0,'defaulttextinterpreter','latex')

% U velocity 
figure()
plot(uMean,zth/lz,'-')
hold all
plot(zeros(size(z_cc)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$u/u_*$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% fluctation variance: sigma2
figure()
plot(sigma2,zth/lz,'-')
hold all
plot(sig2_new,z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\sigma^2/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% dissipation rate (epps)
figure()
plot(epps,zth/lz,'-')
hold all
plot(epps_new,z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\varepsilon\delta/u_*^3$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% tke
figure()
plot((z_cc.*epps_new).^(2/3),z_cc/lz,'o','color',[0.8500 0.3250 0.0980])
ylabel('$z/\delta$') 
xlabel('$tke/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('QES')
