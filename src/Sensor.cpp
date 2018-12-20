#include "Sensor.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <limits>

using namespace std;




void Sensor::inputWindProfile(float dx, float dy, float dz, int nx, int ny, int nz, double *u0, double *v0, double *w0, 
							 int num_sites, int *site_blayer_flag, float *site_one_overL, float *site_xcoord, 
							 float *site_ycoord, float *site_wind_dir, float *site_z0, float *site_z_ref, float *site_U_ref, 
							 float *x, float *y, float *z, Canopy* canopy, float *site_canopy_H, float *site_atten_coeff)
{

	float psi, x_temp, u_star; 
	float rc_sum, rc_val, xc, yc, rc, dn, lamda, s_gamma;
	float sum_wm, sum_wu, sum_wv;
	int iwork = 0, jwork = 0;
	float dxx, dyy, u12, u34, v12, v34;
	const float vk = 0.4;			/// Von Karman's constant
	float canopy_d, u_H;

	std::vector<std::vector<double>> u_prof(num_sites, std::vector<double>(nz,0.0));
	std::vector<std::vector<double>> v_prof(num_sites, std::vector<double>(nz,0.0));
	int icell_face, icell_cent;

	std::vector<float> u0_int(num_sites,0.0);  
	std::vector<float> v0_int(num_sites,0.0); 
	std::vector<float> site_theta(num_sites,0.0); 

	std::vector<std::vector<std::vector<double>>> wm(num_sites, std::vector<std::vector<double>>(nx, std::vector<double>(ny,0.0)));
	std::vector<std::vector<std::vector<double>>> wms(num_sites, std::vector<std::vector<double>>(nx, std::vector<double>(ny,0.0)));

	for (int i = 0 ; i < num_sites; i++)
	{
		site_theta[i] = (270.0-site_wind_dir[i])*M_PI/180.0;
		if (site_blayer_flag[i] == 0)
		{
			for (int k = 1; k < nz; k++)
			{
				u_prof[i][k] = 0.0;
				v_prof[i][k] = 0.0;
			}
		}

		if (site_blayer_flag[i] == 1)
		{
			for (int k = 1; k < nz-1; k++)
			{
				if (k == 1)
				{
					if (site_z_ref[i]*site_one_overL[i] >= 0)
					{
						psi = 4.7*site_z_ref[i]*site_one_overL[i];
					}
					else 
					{
						x_temp = pow((1.0-15.0*site_z_ref[i]*site_one_overL[i]),0.25);
						psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
					}
				
					u_star = site_U_ref[i]*vk/(log((site_z_ref[i]+site_z0[i])/site_z0[i])+psi);
				}
				if (z[k]*site_one_overL[i] >= 0)
				{
					psi = 4.7*z[k]*site_one_overL[i];
				}
				else 
				{
					x_temp = pow((1.0-15.0*z[k]*site_one_overL[i]),0.25);
					psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
				}
	
				u_prof[i][k] = (cos(site_theta[i])*u_star/vk)*(log((z[k]+site_z0[i])/site_z0[i])+psi);
				v_prof[i][k] = (sin(site_theta[i])*u_star/vk)*(log((z[k]+site_z0[i])/site_z0[i])+psi);				
			}
		}

		if (site_blayer_flag[i] == 2)
		{
			for (int k = 1; k < nz; k++)
			{
				u_prof[i][k] = cos(site_theta[i])*site_U_ref[i]*pow((z[k]/site_z_ref[i]),site_z0[i]);
				v_prof[i][k] = sin(site_theta[i])*site_U_ref[i]*pow((z[k]/site_z_ref[i]),site_z0[i]);
			}
		}

		if (site_blayer_flag[i] == 3)
		{
			for (int k = 1; k< nz; k++)
			{
				if (k == 1)
				{
					if (site_z_ref[i]*site_one_overL[i] > 0)
					{
						psi = 4.7*site_z_ref[i]*site_one_overL[i];
					}
					else
					{
						x_temp = pow((1.0-15.0*site_z_ref[i]*site_one_overL[i]),0.25);
						psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
					}
					u_star = site_U_ref[i]*vk/(log(site_z_ref[i]/site_z0[i])+psi);
					canopy_d = canopy->bisection(u_star, site_z0[i], site_canopy_H[i], site_atten_coeff[i], vk, psi);
					if (site_canopy_H[i]*site_one_overL[i] > 0)
					{
						psi = 4.7*(site_canopy_H[i]-canopy_d)*site_one_overL[i];
					}
					else
					{
						x_temp = pow((1.0-15.0*(site_canopy_H[i]-canopy_d)*site_one_overL[i]),0.25);
						psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
					}
					u_H = (u_star/vk)*(log((site_canopy_H[i]-canopy_d)/site_z0[i])+psi);
					if (site_z_ref[i] < site_canopy_H[i])
					{
						site_U_ref[i] /= u_H*exp(site_atten_coeff[i]*(site_z_ref[i]/site_canopy_H[i])-1.0);
					}
					else
					{
						if (site_z_ref[i]*site_one_overL[i] > 0)
						{
							psi = 4.7*(site_z_ref[i]-canopy_d)*site_one_overL[i];
						}
						else
						{
							x_temp = pow(1.0-15.0*(site_z_ref[i]-canopy_d)*site_one_overL[i],0.25);
							psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
						}
						site_U_ref[i] /= ((u_star/vk)*(log((site_z_ref[i]-canopy_d)/site_z0[i])+psi));
					}
					u_star *= site_U_ref[i];
					u_H *= site_U_ref[i];
				}
				
				if (z[k] < site_canopy_H[i])
				{
					u_prof[i][k] = cos(site_theta[i]) * u_H*exp(site_atten_coeff[i]*((z[k]/site_canopy_H[i]) -1.0));
					v_prof[i][k] = sin(site_theta[i]) * u_H*exp(site_atten_coeff[i]*((z[k]/site_canopy_H[i]) -1.0));
				}
				if (z[k] > site_canopy_H[i])
				{
					if (z[k]*site_one_overL[i] > 0)
					{
						psi = 4.7*(z[k]-canopy_d)*site_one_overL[i];
					}
					else
					{
						x_temp = pow(1.0-15.0*(z[k]-canopy_d)*site_one_overL[i],0.25);
						psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
					}
					u_prof[i][k] = (cos(site_theta[i])*u_star/vk)*(log((z[k]-canopy_d)/site_z0[i])+psi);
					v_prof[i][k] = (sin(site_theta[i])*u_star/vk)*(log((z[k]-canopy_d)/site_z0[i])+psi);	
				}
			}
		}

	}


	if (num_sites == 1)
	{
		for ( int k = 0; k < nz; k++)
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
				
					icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values                                
		            u0[icell_face] = u_prof[0][k];
					v0[icell_face] = v_prof[0][k];
					w0[icell_face] = 0.0;         /// Perpendicular wind direction
				}
    	    }
   	    }
   	}
	else 
	{
		rc_sum = 0.0;
		for (int i = 0; i < num_sites; i++)
		{
			rc_val = 1000000.0;
			for (int ii = 0; ii < num_sites; ii++)
			{
				xc = site_xcoord[ii] - site_xcoord[i];
				yc = site_ycoord[ii] - site_ycoord[i];
				rc = sqrt(pow(xc,2.0)+pow(yc,2.0));
				if (rc < rc_val && ii != i){
					rc_val = rc;
				}
			}
			rc_sum = rc_sum+rc_val;
		}
		dn = rc_sum/num_sites;
		lamda = 5.052*pow((2*dn/M_PI),2.0);
		s_gamma = 0.2;
		for (int j=0; j<ny-1; j++)
		{
			for (int i=0; i<nx-1; i++)
			{
				sum_wm = 0.0;
				for (int ii=0; ii<num_sites; ii++)
				{
					wm[ii][i][j] = exp((-1/lamda)*pow(site_xcoord[ii]-x[i],2.0)-(1/lamda)*pow(site_ycoord[ii]-y[j],2.0));
					wms[ii][i][j] = exp((-1/(s_gamma*lamda))*pow(site_xcoord[ii]-x[i],2.0)-(1/(s_gamma*lamda))*
										pow(site_ycoord[ii]-y[j],2.0));
					sum_wm += wm[ii][i][j];
				}
				if (sum_wm == 0)
				{
					for (int ii = 0; ii<num_sites; ii++)
					{
						wm[ii][i][j] = 1e-20;
					}
				}
			}
		}
	
		for (int k=1; k<nz-1; k++)
		{
			for (int j=0; j<ny-1; j++)
			{
				for (int i=0; i<nx-1; i++)
				{
					sum_wu = 0.0;
					sum_wv = 0.0;
					sum_wm = 0.0;
					for (int ii=0; ii<num_sites; ii++)
					{
						sum_wu += wm[ii][i][j]*u_prof[ii][k];
						sum_wv += wm[ii][i][j]*v_prof[ii][k];	
						sum_wm += wm[ii][i][j];
					}
					icell_face = i + j*nx + k*nx*ny;
					u0[icell_face] = sum_wu/sum_wm;
					v0[icell_face] = sum_wv/sum_wm;	
					w0[icell_face] = 0.0;
				}
			}
	
			for (int ii=0; ii<num_sites; ii++)
			{
				if(site_xcoord[ii]>0 && site_xcoord[ii]<(nx-1)*dx && site_ycoord[ii]>0 && site_ycoord[ii]<(ny-1)*dy)
				{
					for (int j=0; j<ny-1; j++)
					{
						if (y[j]<site_ycoord[ii])
						{
							jwork = j;
						}
					}
					for (int i=0; i<nx-1; i++)
					{
						if (x[i]<site_xcoord[ii])
						{
							iwork = i;
						}
					} 
					dxx = site_xcoord[ii]-x[iwork];
					dyy = site_ycoord[ii]-y[jwork];
					int index_work = iwork+jwork*nx+k*nx*ny;
					u12 = (1-(dxx/dx))*u0[index_work+nx]+(dxx/dx)*u0[index_work+1+nx];
					u34 = (1-(dxx/dx))*u0[index_work]+(dxx/dx)*u0[index_work+1];
					u0_int[ii] = (dyy/dy)*u12+(1-(dyy/dy))*u34;
		
					v12 = (1-(dxx/dx))*v0[index_work+nx]+(dxx/dx)*v0[index_work+1+nx];
					v34 = (1-(dxx/dx))*v0[index_work]+(dxx/dx)*v0[index_work+1];
					v0_int[ii] = (dyy/dy)*v12+(1-(dyy/dy))*v34;
				}
				else
				{
					u0_int[ii] = u_prof[ii][k];
					v0_int[ii] = v_prof[ii][k];
				}
			}
	
			for (int j=0; j<ny-1; j++)
			{
				for (int i=0; i<nx-1; i++)
				{
					sum_wu = 0.0;
					sum_wv = 0.0;
					sum_wm = 0.0;
					for (int ii=0; ii<num_sites; ii++)
					{
						sum_wu += wm[ii][i][j]*(u_prof[ii][k]-u0_int[ii]);
						sum_wv += wm[ii][i][j]*(v_prof[ii][k]-v0_int[ii]);	
						sum_wm += wm[ii][i][j];
					}
					if (sum_wm != 0)
					{
						icell_face = i + j*nx + k*nx*ny;
						u0[icell_face] = u0[icell_face]+sum_wu/sum_wm;
						v0[icell_face] = v0[icell_face]+sum_wv/sum_wm;
					}	
				}
			}
		}
	}

}

