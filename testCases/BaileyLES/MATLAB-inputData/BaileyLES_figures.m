%% ========================================================================
% some figure to vizualize the data generated vs original data:
set(0,'defaulttextinterpreter','latex')

% U velocity 
figure()
plot(squeeze(mean(mean(u,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(u_out(:,1:end-1,1:end-1),1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$u/u_*$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% V velocity 
figure()
plot(squeeze(mean(mean(v,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(v_out(1:end-1,:,1:end-1),1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$v/u_*$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% W velocity 
figure()
plot(squeeze(mean(mean(w,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(w_out(1:end-1,1:end-1,:),1),2)),z_pp/lz,'o')
ylabel('$z/\delta$')
xlabel('$w/u_*$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

%% ========================================================================
% some figure to vizualize the data generated vs original data:
set(0,'defaulttextinterpreter','latex')

% tau11 stress 
figure()
plot(squeeze(mean(mean(txx,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau11,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{xx}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% tau22 stress 
figure()
plot(squeeze(mean(mean(tyy,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau22,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{yy}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% tau33 stress 
figure()
plot(squeeze(mean(mean(tzz,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau33,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{zz}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')


% tau12 stress 
figure()
plot(squeeze(mean(mean(txy,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau12,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{xy}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% tau13 stress 
figure()
plot(squeeze(mean(mean(txz,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau13,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{xz}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% tau23 stress 
figure()
plot(squeeze(mean(mean(tyz,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(tau23,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\tau_{yz}/u_*^2$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')

% epps dissipation rate 
figure()
plot(squeeze(mean(mean(epps,1),2)),z/lz,'x')
hold all
plot(squeeze(mean(mean(epps_out,1),2)),z_cc/lz,'o')
ylabel('$z/\delta$')
xlabel('$\epsilon$')
grid on
ylim([0-dz lz+dz]/lz)
legend('data','QES')
