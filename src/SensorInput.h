
#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <iostream>

class SensorInput
{

private:
	int num_sites;
	int *site_blayer_flag, *site_coord_flag;
	float *site_one_overL, *site_xcoord, *site_ycoord;
	double **u_prof, **v_prof;
	float *site_wind_dir, *site_z0, *site_z_ref, *site_U_ref, *site_theta;
	float psi, x_temp, u_star; 
	float rc_sum, rc_value, xc, yc, rc, dn, lamda, s_gamma;
	float sum_wm, sum_wu, sum_wv;
	int iwork, jwork,rc_val;
	float dxx, dyy, u12, u34, v12, v34;
	float *u0_int, *v0_int;
	
