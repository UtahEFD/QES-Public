
%%% copy and paste the variables that are output from CUDA-Plume when a
%%% particle goes rogue

dt = 2.22
C0 = 4.0
invariantTol = 10e-5


uFluct = 15.6031
vFluct = -35.6785
wFluct = 1.73429
xPos = 4591.84
yPos = 874.4
zPos = 974.92
uFluct_old = -0.108941
vFluct_old = 0.123749
wFluct_old = -0.0708372
txx_old = 0.00619544
txy_old = 0.00857066
txz_old = -0.000896642
tyy_old = -0.00955689
tyz_old = 0.00305634
tzz_old = 0.0521669
CoEps = 0.000170676
uMean = 11.4109
vMean = -0.0360971
wMean = 0.0539727
txx_before = 0.00619544
txy_before = 0.00857066
txz_before = -0.000896642
tyy_before = -0.00955689
tyz_before = 0.00305634
tzz_before = 0.0521669
flux_div_x = -8.99155e-05
flux_div_y = 0.000143699
flux_div_z = 0.00131714
txx = 0.0277318
txy = 0.00857066
txz = -0.000896642
tyy = 0.0119794
tyz = 0.00305634
tzz = 0.0737032
lxx = 36.0597
lxy = -25.7989
lxz = 1.50852
lyy = 83.4764
lyz = -3.46162
lzz = 13.5679
xRandn = -0.251173
yRandn = 1.34587
zRandn = -0.0405581
dtxxdt = 0.00970105
dtxydt = 0
dtxzdt = 0
dtyydt = 0.00970105
dtyzdt = 0
dtzzdt = 0.00970105
A_11 = -0.618534
A_12 = -0.272919
A_13 = 0.0159582
A_21 = -0.272919
A_22 = -0.116927
A_23 = -0.0366195
A_31 = 0.0159582
A_32 = -0.0366195
A_33 = -0.856469
b_11 = 0.11393
b_21 = -0.150106
b_31 = 0.0701647
A_11_inv = 32.6182
A_12_inv = -77.3601
A_13_inv = 3.9154
A_21_inv = -77.3601
A_22_inv = 174.805
A_23_inv = -8.91545
A_31_inv = 3.9154
A_32_inv = -8.91545
A_33_inv = -0.713439


%%% copy and paste the calculation of the A and b matrices done in
%%% CUDA-Plume, adjusting the output name to have an additional test_ on
%%% it.
%%% Compare test_A and test_b values with A and b values of the
%%% CUDA-Plume output
test_A_11 = -1.0 + 0.50*(-CoEps*lxx + lxx*dtxxdt + lxy*dtxydt + lxz*dtxzdt)*dt
test_A_12 =        0.50*(-CoEps*lxy + lxy*dtxxdt + lyy*dtxydt + lyz*dtxzdt)*dt
test_A_13 =        0.50*(-CoEps*lxz + lxz*dtxxdt + lyz*dtxydt + lzz*dtxzdt)*dt

test_A_21 =        0.50*(-CoEps*lxy + lxx*dtxydt + lxy*dtyydt + lxz*dtyzdt)*dt
test_A_22 = -1.0 + 0.50*(-CoEps*lyy + lxy*dtxydt + lyy*dtyydt + lyz*dtyzdt)*dt
test_A_23 =        0.50*(-CoEps*lyz + lxz*dtxydt + lyz*dtyydt + lzz*dtyzdt)*dt

test_A_31 =        0.50*(-CoEps*lxz + lxx*dtxzdt + lxy*dtyzdt + lxz*dtzzdt)*dt
test_A_32 =        0.50*(-CoEps*lyz + lxy*dtxzdt + lyy*dtyzdt + lyz*dtzzdt)*dt
test_A_33 = -1.0 + 0.50*(-CoEps*lzz + lxz*dtxzdt + lyz*dtyzdt + lzz*dtzzdt)*dt


test_b_11 = -uFluct_old - 0.50*flux_div_x*dt - sqrt(CoEps*dt)*xRandn
test_b_21 = -vFluct_old - 0.50*flux_div_y*dt - sqrt(CoEps*dt)*yRandn
test_b_31 = -wFluct_old - 0.50*flux_div_z*dt - sqrt(CoEps*dt)*zRandn


%%% if the test_A matrix match the CUDA-Plume A matrix, copy and paste the 
%%% invert3 functions from CUDA-Plume, and call the output test_Ainv. 
%%% Since test_A is the same as CUDA-Plume A, can just use the 
%%% CUDA-Plume values for the calculation.
%%% Compare test_Ainv with the A_inv values of the CUDA-Plume output
det = A_11*(A_22*A_33 - A_23*A_32) - A_12*(A_21*A_33 - A_23*A_31) + A_13*(A_21*A_32 - A_22*A_31)

test_Ainv_11 =  (A_22*A_33 - A_23*A_32)/det
test_Ainv_12 = -(A_12*A_33 - A_13*A_32)/det
test_Ainv_13 =  (A_12*A_23 - A_22*A_13)/det
test_Ainv_21 = -(A_21*A_33 - A_23*A_31)/det
test_Ainv_22 =  (A_11*A_33 - A_13*A_31)/det
test_Ainv_23 = -(A_11*A_23 - A_13*A_21)/det
test_Ainv_31 =  (A_21*A_32 - A_31*A_22)/det
test_Ainv_32 = -(A_11*A_32 - A_12*A_31)/det
test_Ainv_33 =  (A_11*A_22 - A_12*A_21)/det

%%% if the test_Ainv and test_b matrices match the CUDA-Plume Ainv and b
%%% matrices, copy and paste the matmult function from CUDA-Plume, and
%%% call the output x. Since test_Ainv and test_b are the same as
%%% CUDA-Plume Ainv and b, can just use the CUDA-Plume values for the
%%% calculation.
%%% Compare x with the velFluct values of the CUDA-Plume output
x_11 = b_11*A_11_inv + b_21*A_12_inv + b_31*A_13_inv
x_21 = b_11*A_21_inv + b_21*A_22_inv + b_31*A_23_inv
x_31 = b_11*A_31_inv + b_21*A_32_inv + b_31*A_33_inv



%%% now do to the same thing, but using code that has worked in the past.
%%% In this case, I took the matlab code for the different functions and
%%% changed the variable names to match the CUDA-Plume output variables. So
%%% I used mat_ instead of test_. Could use Bail_ instead of test_ for
%%% Bailey code comparisons.



%%% copy and paste the calculation of the A and b matrices done in
%%% the past matlab code, adjusting the output name to have an additional 
%%% mat_ on it.
%%% Compare mat_A and mat_b values with A and b values of the
%%% CUDA-Plume output
mat_A = zeros(3,3);
mat_b = zeros(3,1);
mat_A(1,1) = -1 + 0.5*(-CoEps*lxx + lxx*dtxxdt + lxy*dtxydt + lxz*dtxzdt)*dt;
mat_A(1,2) =      0.5*(-CoEps*lxy + lxy*dtxxdt + lyy*dtxydt + lyz*dtxzdt)*dt;
mat_A(1,3) =      0.5*(-CoEps*lxz + lxz*dtxxdt + lyz*dtxydt + lzz*dtxzdt)*dt;

mat_A(2,1) =      0.5*(-CoEps*lxy + lxx*dtxydt + lxy*dtyydt + lxz*dtyzdt)*dt;
mat_A(2,2) = -1 + 0.5*(-CoEps*lyy + lxy*dtxydt + lyy*dtyydt + lyz*dtyzdt)*dt;
mat_A(2,3) =      0.5*(-CoEps*lyz + lxz*dtxydt + lyz*dtyydt + lzz*dtyzdt)*dt;

mat_A(3,1) =      0.5*(-CoEps*lxz + lxx*dtxzdt + lxy*dtyzdt + lxz*dtzzdt)*dt;
mat_A(3,2) =      0.5*(-CoEps*lyz + lxy*dtxzdt + lyy*dtyzdt + lyz*dtzzdt)*dt;
mat_A(3,3) = -1 + 0.5*(-CoEps*lzz + lxz*dtxzdt + lyz*dtyzdt + lzz*dtzzdt)*dt;

mat_b(1) = -uFluct_old - 0.5*flux_div_x*dt - sqrt(CoEps*dt)*xRandn;
mat_b(2) = -vFluct_old - 0.5*flux_div_x*dt - sqrt(CoEps*dt)*yRandn;
mat_b(3) = -wFluct_old - 0.5*flux_div_x*dt - sqrt(CoEps*dt)*zRandn;

% now display so can compare values
mat_A
mat_b


%%% if the mat_A and mat_b matrices match the CUDA-Plume A and b
%%% matrices, copy and paste the solution method from the past matlab code,
%%% and call the output mat_velFluct. Since test_A and test_b are the same 
%%% as CUDA-Plume A and b, can just use the CUDA-Plume values for the
%%% calculation.
%%% Compare mat_velFluct with the velFluct values of the CUDA-Plume output
mat_A = [A_11,A_12,A_13;A_21,A_22,A_23;A_31,A_32,A_33]
mat_b = [b_11;b_21;b_31]

mat_velFluct = mat_A\mat_b


%%% because all the tests described above worked for me, and the values
%%% were as expected, this means that the invert3, matmult, and A and b
%%% formulation functions in CUDA-Plume are working correctly, at least for
%%% the Ax=b calculation (still need to check invert3 for inverting tao).
%%% This means that the problem has to be before the A and b formulation,
%%% or that the problem has to do with what is getting fed into the lines
%%% of the A and b formulation. Narrows things down quite a bit, but at the
%%% same time not really.




%%% this next step is to make sure that, for a given position, the interp3D
%%% functions are grabbing the same values in both codes. It isn't as
%%% straight forwards as the other one, cause it means having access to the
%%% desired interp3D function, and defining the cell grids the same, so
%%% watch out for these pitfalls

%%% to do this, I ran the old matlab code, pausing at the end of the first
%%% iteration, for the same case as done in the CUDA-Plume code (LES
%%% heterogeneous anisotropic), with the gradients calculated using the
%%% same method (2nd order forward and backward differencing). Then after
%%% that pause, since I had the same cell grid as well, I copied and pasted
%%% the interp3D calls to the datasets I have for CUDA-Plume debug output,
%%% but replacing the current positions with the positions I am using here.
%%% Then after editing, I copied and pasted the desired lines of code into
%%% the command window and did comparisons.
%%% Because this will cause me trouble running other things, I've included
%%% the interp3D lines of code here in comments

%%% compare mat_uMean with CUDA-Plume uMean
%%% compare mat_tao_before with CUDA-Plume tao_before
%%% compare mat_flux_div_dir with CUDA-Plume flux_div_dir
%%% compare mat_CoEps with CUDA-Plume CoEps, but make sure the C0 is the
%%% same for both before this

% % % mat_uMean = interp3D(uMean_data,'u',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % mat_vMean = interp3D(vMean_data,'v',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % mat_wMean = interp3D(wMean_data,'w',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % mat_CoEps = C0*interp3D(epps_data,'epps',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % 
% % % mat_txx_before = interp3D(txx_data,'txx',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % mat_txy_before = interp3D(txy_data,'txy',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % mat_txz_before = interp3D(txz_data,'txz',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % mat_tyy_before = interp3D(tyy_data,'tyy',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % mat_tyz_before = interp3D(tyz_data,'tyz',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % mat_tzz_before = interp3D(tzz_data,'tzz',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % 
% % % mat_flux_div_x = interp3D(flux_div_x_data,'flux_div_x',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % mat_flux_div_y = interp3D(flux_div_y_data,'flux_div_y',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)
% % % mat_flux_div_z = interp3D(flux_div_z_data,'flux_div_z',xPos,yPos,zPos,cellGrid_dx,cellGrid_dy,cellGrid_dz,x_nCells,y_nCells,z_nCells)

%%% sweeeeet!!! Looks like the interp3D came out with the same values, even
%%% for tao_before and flux_div, at least to the fourth decimal places. This means
%%% that the problem is not in the interp3D functions, but in how the
%%% interpolated data is handled leading up to the throwing stuff into the
%%% A and b matrices. 
%%% This means the problem is either the inverse matrix
%%% calculations, the dtaodt time derivative or change in tao between
%%% iterations, or the random number generation.
%%% it is not so easy to test the dtaodt time derivative of the stress
%%% tensor, or the random number generation, but the tensor inverse can be
%%% checked here.


%%% makes sense to just try to do like I did with the interp3D functions.
%%% So I need to run the old matlab code, pausing at the end of the first
%%% iteration, for the same case as done in the CUDA-Plume code (LES
%%% heterogeneous anisotropic), with the gradients calculated using the
%%% same method (2nd order forward and backward differencing).
%%% Then after that pause, to run copies of the matlab makeRealizable and
%%% invert3 on the CUDA-Plume debug output, comparing the results with
%%% CUDA-Plume debug output.
%%% going to have to copy and paste these matlab functions, edit them, then
%%% copy into the command window for the comparisons.
%%% Then after editing, I copied and pasted the desired lines of code into
%%% the command window and did comparisons.
%%% Because this will cause me trouble running other things, I've included
%%% the makeRealizeable and interp3 matlab lines of code here in comments


%%% notice that makeTaoRealizable uses tao_before as input to get tao out
%%% this works because tao_before is confirmed to be the correct value
%%% make sure invarianceTol is the same as the CUDA-Plume
%%% compare mat_tao with CUDA-Plume tao (not tao_before)
% % % [mat_txx,mat_txy,mat_txz,mat_tyy,mat_tyz,mat_tzz] = makeTaoRealizable(txx_before,txy_before,txz_before,tyy_before,tyz_before,tzz_before,invariantTol)


%%% notice that makeTaoRealizable uses tao as input to get lao out because
%%% tao is assumed to be the same as input
%%% compare mat_lao (txx now lxx, inverse tao) with CUDA-Plume lao
% % % makeRealizable = true;
% % % [mat_lxx,mat_lxy,mat_lxz,mat_lyy,mat_lyz,mat_lzz] = invertStressTensor(txx,txy,txz,tyy,tyz,tzz,makeRealizable)



%%% woah, it looks like makeTaoRealizable gets the same values as the
%%% CUDA-Plume data for the off diagonals of tao, but has completely
%%% differnt values for the diagonals of tao. So something is definitely
%%% going on in makeRealizeable.

%%% fortunately it looks like invert3 is still working correctly. The lao
%%% values matched between the matlab and CUDA-Plume code. Now I just have
%%% to figure out what is wrong with makeRealizable



