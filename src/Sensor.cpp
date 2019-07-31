/*
 * CUDA-URB
 * Copyright (c) 2019 Behnam Bozorgmehr
 * Copyright (c) 2019 Eric Pardyjak
 * Copyright (c) 2019 Rob Stoll
 * Copyright (c) 2019 Pete Willemsen
 *
 * This file is part of CUDA-URB
 *
 * MIT License
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */


#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <limits>

#include "Sensor.h"

#include "URBInputData.h"
#include "URBGeneralData.h"


using namespace std;




void Sensor::inputWindProfile(const URBInputData *UID, URBGeneralData *ugd)
{
    // replicate some local variable to make reference to URBInputData
    // and URBGeneralData elements to make reading code below more conciser
    int dx = ugd->dx;
    int dy = ugd->dy;
    int dz = ugd->dz;
    int nx = ugd->nx;
    int ny = ugd->ny;
    int nz = ugd->nz;
    std::vector<double> &u0 = ugd->u0;
    std::vector<double> &v0 = ugd->v0;
    std::vector<double> &w0 = ugd->w0;

    const std::vector<float> &z = ugd->z;
    
    std::vector<Sensor*> &sensors = UID->metParams->sensors;

    const float UTMx = UID->simParams->UTMx;
    const float UTMy = UID->simParams->UTMy;
    const float UTMZone = UID->simParams->UTMZone;

    const float theta = ugd->theta;

    const std::vector<float> &z0_domain = ugd->z0_domain;
    const std::vector<int> &terrain_id = ugd->terrain_id;
    const std::vector<double> &terrain = ugd->terrain;
    const int z0_domain_flag = UID->metParams->z0_domain_flag;
    


	float psi, psi_first, x_temp, u_star;
	float rc_sum, rc_val, xc, yc, rc, dn, lamda, s_gamma;
	float sum_wm, sum_wu, sum_wv;
	int iwork = 0, jwork = 0;
	float dxx, dyy, u12, u34, v12, v34;
	const float vk = 0.4;			/// Von Karman's constant
	float canopy_d, u_H;
	float site_UTM_x, site_UTM_y;
	float site_lon, site_lat;
	float wind_dir, z0_new, z0_high, z0_low;
	float u_new, u_new_low, u_new_high;
	int log_flag, iter, id;
	float a1, a2, a3;
	float site_mag;
	float blending_height = 0.0, average__one_overL = 0.0;
	int max_terrain = 1;
	std::vector<float> x,y;

	int num_sites = sensors.size();
	std::vector<std::vector<double>> u_prof(num_sites, std::vector<double>(nz,0.0));
	std::vector<std::vector<double>> v_prof(num_sites, std::vector<double>(nz,0.0));
	int icell_face, icell_cent;

	std::vector<int> site_i(num_sites,0);
	std::vector<int> site_j(num_sites,0);
	std::vector<int> site_id(num_sites,0);
	std::vector<float> u0_int(num_sites,0.0);
	std::vector<float> v0_int(num_sites,0.0);
	std::vector<float> site_theta(num_sites,0.0);

	std::vector<std::vector<std::vector<double>>> wm(num_sites, std::vector<std::vector<double>>(nx, std::vector<double>(ny,0.0)));
	std::vector<std::vector<std::vector<double>>> wms(num_sites, std::vector<std::vector<double>>(nx, std::vector<double>(ny,0.0)));

	// Loop through all sites and create velocity profiles (u0,v0)
	for (auto i = 0 ; i < num_sites; i++)
	{
		float convergence = 0.0;
	  site_i[i] = sensors[i]->site_xcoord/dx;
		site_j[i] = sensors[i]->site_ycoord/dy;
		site_id[i] = site_i[i] + site_j[i]*(nx-1);
		for (auto j=0; j<sensors[i]->site_z_ref.size(); j++)
		{
			sensors[i]->site_z_ref[j] -= terrain[site_id[i]];
		}
		int id = 1;
		int counter = 0;
		if (sensors[i]->site_z_ref[0] > 0)
		{
			blending_height += sensors[i]->site_z_ref[0]/num_sites;
		}
		else
		{
			if (sensors[i]->site_blayer_flag == 4 )
			{
				while (id<sensors[i]->site_z_ref.size() && sensors[i]->site_z_ref[id]>0 && counter<1)
				{
					blending_height += sensors[i]->site_z_ref[id]/num_sites;
					counter += 1;
					id += 1;
				}
			}
		}

		std::cout << "blending_height:   " << blending_height << std::endl;
		average__one_overL += sensors[i]->site_one_overL/num_sites;
		if (UTMx != 0 && UTMy != 0)
		{
			if (sensors[i]->site_coord_flag == 1)
			{
				sensors[i]->site_UTM_x = sensors[i]->site_xcoord * acos(theta) + sensors[i]->site_ycoord * asin(theta) + UTMx;
				sensors[i]->site_UTM_y = sensors[i]->site_xcoord * asin(theta) + sensors[i]->site_ycoord * acos(theta) + UTMy;
				sensors[i]->site_UTM_zone = UTMZone;
				// Calling UTMConverter function to convert UTM coordinate to lat/lon and vice versa (located in Sensor.cpp)
				UTMConverter (sensors[i]->site_lon, sensors[i]->site_lat, sensors[i]->site_UTM_x, sensors[i]->site_UTM_y, sensors[i]->site_UTM_zone, 1);
			}

			if (sensors[i]->site_coord_flag == 2)
			{
				// Calling UTMConverter function to convert UTM coordinate to lat/lon and vice versa (located in Sensor.cpp)
				UTMConverter (sensors[i]->site_lon, sensors[i]->site_lat, sensors[i]->site_UTM_x, sensors[i]->site_UTM_y, sensors[i]->site_UTM_zone, 1);
			}

			getConvergence(sensors[i]->site_lon, sensors[i]->site_lat, sensors[i]->site_UTM_zone, convergence);
		}

		site_theta[i] = (270.0-sensors[i]->site_wind_dir[0])*M_PI/180.0;

		// If site has a uniform velocity profile
		if (sensors[i]->site_blayer_flag == 0)
		{
			for (auto k = terrain_id[site_id[i]]; k < nz; k++)
			{
				u_prof[i][k] = cos(site_theta[i])*sensors[i]->site_U_ref[0];
				v_prof[i][k] = sin(site_theta[i])*sensors[i]->site_U_ref[0];
			}
		}
		// Logarithmic velocity profile
		if (sensors[i]->site_blayer_flag == 1)
		{
			for (auto k = terrain_id[site_id[i]]; k < nz; k++)
			{
				if (k == terrain_id[site_id[i]])
				{
					if (sensors[i]->site_z_ref[0]*sensors[i]->site_one_overL >= 0)
					{
						psi = 4.7*sensors[i]->site_z_ref[0]*sensors[i]->site_one_overL;
					}
					else
					{
						x_temp = pow((1.0-15.0*sensors[i]->site_z_ref[0]*sensors[i]->site_one_overL),0.25);
						psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
					}

					u_star = sensors[i]->site_U_ref[0]*vk/(log((sensors[i]->site_z_ref[0]+sensors[i]->site_z0)/sensors[i]->site_z0)+psi);
				}
				if (z[k]*sensors[i]->site_one_overL >= 0)
				{
					psi = 4.7*z[k]*sensors[i]->site_one_overL;
				}
				else
				{
					x_temp = pow((1.0-15.0*z[k]*sensors[i]->site_one_overL),0.25);
					psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
				}

				u_prof[i][k] = (cos(site_theta[i])*u_star/vk)*(log((z[k]+sensors[i]->site_z0)/sensors[i]->site_z0)+psi);
				v_prof[i][k] = (sin(site_theta[i])*u_star/vk)*(log((z[k]+sensors[i]->site_z0)/sensors[i]->site_z0)+psi);
			}
		}

		// Exponential velocity profile
		if (sensors[i]->site_blayer_flag == 2)
		{
			for (auto k = terrain_id[site_id[i]]; k < nz; k++)
			{
				u_prof[i][k] = cos(site_theta[i])*sensors[i]->site_U_ref[0]*pow((z[k]/sensors[i]->site_z_ref[0]),sensors[i]->site_z0);
				v_prof[i][k] = sin(site_theta[i])*sensors[i]->site_U_ref[0]*pow((z[k]/sensors[i]->site_z_ref[0]),sensors[i]->site_z0);
			}
		}

		// Canopy velocity profile
		if (sensors[i]->site_blayer_flag == 3)
		{
			for (auto k = terrain_id[site_id[i]]; k< nz; k++)
			{
				if (k == terrain_id[site_id[i]])
				{
					if (sensors[i]->site_z_ref[0]*sensors[i]->site_one_overL > 0)
					{
						psi = 4.7*sensors[i]->site_z_ref[0]*sensors[i]->site_one_overL;
					}
					else
					{
						x_temp = pow((1.0-15.0*sensors[i]->site_z_ref[0]*sensors[i]->site_one_overL),0.25);
						psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
					}
					u_star = sensors[i]->site_U_ref[0]*vk/(log(sensors[i]->site_z_ref[0]/sensors[i]->site_z0)+psi);
					canopy_d = ugd->canopyBisection(u_star, sensors[i]->site_z0, sensors[i]->site_canopy_H, sensors[i]->site_atten_coeff, vk, psi);
					if (sensors[i]->site_canopy_H*sensors[i]->site_one_overL > 0)
					{
						psi = 4.7*(sensors[i]->site_canopy_H-canopy_d)*sensors[i]->site_one_overL;
					}
					else
					{
						x_temp = pow((1.0-15.0*(sensors[i]->site_canopy_H-canopy_d)*sensors[i]->site_one_overL),0.25);
						psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
					}
					u_H = (u_star/vk)*(log((sensors[i]->site_canopy_H-canopy_d)/sensors[i]->site_z0)+psi);
					if (sensors[i]->site_z_ref[0] < sensors[i]->site_canopy_H)
					{
						sensors[i]->site_U_ref[0] /= u_H*exp(sensors[i]->site_atten_coeff*(sensors[i]->site_z_ref[0]/sensors[i]->site_canopy_H)-1.0);
					}
					else
					{
						if (sensors[i]->site_z_ref[0]*sensors[i]->site_one_overL > 0)
						{
							psi = 4.7*(sensors[i]->site_z_ref[0]-canopy_d)*sensors[i]->site_one_overL;
						}
						else
						{
							x_temp = pow(1.0-15.0*(sensors[i]->site_z_ref[0]-canopy_d)*sensors[i]->site_one_overL,0.25);
							psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
						}
						sensors[i]->site_U_ref[0] /= ((u_star/vk)*(log((sensors[i]->site_z_ref[0]-canopy_d)/sensors[i]->site_z0)+psi));
					}
					u_star *= sensors[i]->site_U_ref[0];
					u_H *= sensors[i]->site_U_ref[0];
				}

				if (z[k] < sensors[i]->site_canopy_H)
				{
					u_prof[i][k] = cos(site_theta[i]) * u_H*exp(sensors[i]->site_atten_coeff*((z[k]/sensors[i]->site_canopy_H) -1.0));
					v_prof[i][k] = sin(site_theta[i]) * u_H*exp(sensors[i]->site_atten_coeff*((z[k]/sensors[i]->site_canopy_H) -1.0));
				}
				if (z[k] > sensors[i]->site_canopy_H)
				{
					if (z[k]*sensors[i]->site_one_overL > 0)
					{
						psi = 4.7*(z[k]-canopy_d)*sensors[i]->site_one_overL;
					}
					else
					{
						x_temp = pow(1.0-15.0*(z[k]-canopy_d)*sensors[i]->site_one_overL,0.25);
						psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
					}
					u_prof[i][k] = (cos(site_theta[i])*u_star/vk)*(log((z[k]-canopy_d)/sensors[i]->site_z0)+psi);
					v_prof[i][k] = (sin(site_theta[i])*u_star/vk)*(log((z[k]-canopy_d)/sensors[i]->site_z0)+psi);
				}
			}
		}

		// Data entry profile (WRF output)
		if (sensors[i]->site_blayer_flag == 4)
		{
			int z_size = sensors[i]->site_z_ref.size();
			int ii = -1;
			site_theta[i] = (270.0-sensors[i]->site_wind_dir[0])*M_PI/180.0;
			for (auto k=terrain_id[site_id[i]]; k<nz; k++)
			{
				if (z[k] < sensors[i]->site_z_ref[0] || z_size == 1)
				{
					u_prof[i][k] = (sensors[i]->site_U_ref[0]*cos(site_theta[i])/log((sensors[i]->site_z_ref[0]+sensors[i]->site_z0)/sensors[i]->site_z0))
													*log((z[k]+sensors[i]->site_z0)/sensors[i]->site_z0);
					v_prof[i][k] = (sensors[i]->site_U_ref[0]*sin(site_theta[i])/log((sensors[i]->site_z_ref[0]+sensors[i]->site_z0)/sensors[i]->site_z0))
													*log((z[k]+sensors[i]->site_z0)/sensors[i]->site_z0);
				}
				else
				{

					if ( (ii < z_size-2) && (z[k] >= sensors[i]->site_z_ref[ii+1]))
					{
						ii += 1;
						if (abs(sensors[i]->site_wind_dir[ii+1]-sensors[i]->site_wind_dir[ii]) > 180.0)
						{
							if (sensors[i]->site_wind_dir[ii+1] > sensors[i]->site_wind_dir[ii])
							{
								wind_dir = (sensors[i]->site_wind_dir[ii+1]-360.0-sensors[i]->site_wind_dir[ii+1])
														/(sensors[i]->site_z_ref[ii+1]-sensors[i]->site_z_ref[ii]);
							}
							else
							{
								wind_dir = (sensors[i]->site_wind_dir[ii+1]+360.0-sensors[i]->site_wind_dir[ii+1])
														/(sensors[i]->site_z_ref[ii+1]-sensors[i]->site_z_ref[ii]);
							}
						}
						else
						{
							wind_dir = (sensors[i]->site_wind_dir[ii+1]-sensors[i]->site_wind_dir[ii])
													/(sensors[i]->site_z_ref[ii+1]-sensors[i]->site_z_ref[ii]);
						}
						z0_high = 20.0;
						u_star = vk*sensors[i]->site_U_ref[ii]/log((sensors[i]->site_z_ref[ii]+z0_high)/z0_high);
						u_new_high = (u_star/vk)*log((sensors[i]->site_z_ref[ii]+z0_high)/z0_high);
						z0_low = 1e-9;
						u_star = vk*sensors[i]->site_U_ref[ii]/log((sensors[i]->site_z_ref[ii]+z0_low)/z0_low);
						u_new_low = (u_star/vk)*log((sensors[i]->site_z_ref[ii+1]+z0_low)/z0_low);

						if (sensors[i]->site_U_ref[ii+1] > u_new_low && sensors[i]->site_U_ref[ii+1] < u_new_high)
						{
							log_flag = 1;
							iter = 0;
							u_star = vk*sensors[i]->site_U_ref[ii]/log((sensors[i]->site_z_ref[ii]+sensors[i]->site_z0)/sensors[i]->site_z0);
							u_new = (u_star/vk)*log((sensors[i]->site_z_ref[ii+1]+sensors[i]->site_z0)/sensors[i]->site_z0);
							while (iter < 200 && abs(u_new-sensors[i]->site_U_ref[ii]) > 0.0001*sensors[i]->site_U_ref[ii])
							{
								iter += 1;
								z0_new = 0.5*(z0_low+z0_high);
								u_star = vk*sensors[i]->site_U_ref[ii]/log((sensors[i]->site_z_ref[ii]+z0_new)/z0_new);
								u_new = (u_star/vk)*log((sensors[i]->site_z_ref[ii+1]+z0_new)/z0_new);
								if (u_new > sensors[i]->site_z_ref[ii+1])
								{
									z0_high = z0_new;
								}
								else
								{
									z0_low = z0_new;
								}
							}
						}
						else
						{
							log_flag = 0;
							if (ii < z_size-2)
							{
								a1 = ((sensors[i]->site_z_ref[ii+1]-sensors[i]->site_z_ref[ii])
									 	*(sensors[i]->site_U_ref[ii+2]-sensors[i]->site_U_ref[ii])
									 	+(sensors[i]->site_z_ref[ii]-sensors[i]->site_z_ref[ii+2])
									 	*(sensors[i]->site_U_ref[ii+1]-sensors[i]->site_U_ref[ii]))
									 	/((sensors[i]->site_z_ref[ii+1]-sensors[i]->site_z_ref[ii])
									 	*(pow(sensors[i]->site_z_ref[ii+2],2.0)-pow(sensors[i]->site_z_ref[ii],2.0))
									 	+(pow(sensors[i]->site_z_ref[ii+1],2.0)-pow(sensors[i]->site_z_ref[ii],2.0))
									 	*(sensors[i]->site_z_ref[ii]-sensors[i]->site_z_ref[ii+2]));
							}
							else
							{
								a1 = 0.0;
							}
							a2 = ((sensors[i]->site_U_ref[ii+1]-sensors[i]->site_U_ref[ii])
								 	-a1*(pow(sensors[i]->site_z_ref[ii+1],2.0)-pow(sensors[i]->site_z_ref[ii],2.0)))
								 	/(sensors[i]->site_z_ref[ii+1]-sensors[i]->site_z_ref[ii]);
							a3 = sensors[i]->site_U_ref[ii]-a1*pow(sensors[i]->site_z_ref[ii],2.0)
								 	-a2*sensors[i]->site_z_ref[ii];
						}
					}
					if (log_flag == 1)
					{
						site_mag = (u_star/vk)*log((z[k]+z0_new)/z0_new);
					}
					else
					{
						site_mag = a1*pow(z[k], 2.0)+a2*z[k]+a3;
					}
					site_theta[i] = (270.0-(sensors[i]->site_wind_dir[ii]+
											wind_dir*(z[k]-sensors[i]->site_z_ref[ii])))*M_PI/180.0;
					u_prof[i][k] = site_mag*cos(site_theta[i]);
					v_prof[i][k] = site_mag*sin(site_theta[i]);
				}
			}
		}
	}

	x.resize( nx );
	for (size_t i=0; i<nx; i++)
	{
		x[i] = (i-0.5)*dx;          /**< Location of face centers in x-dir */
	}

	y.resize( ny );
	for (auto j=0; j<ny; j++)
	{
		y[j] = (j-0.5)*dy;          /**< Location of face centers in y-dir */
	}


	if (num_sites == 1)
	{
		for ( auto k = 0; k < nz; k++)
		{
			for (auto j = 0; j < ny; j++)
			{
				for (auto i = 0; i < nx; i++)
				{

					icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values
		      u0[icell_face] = u_prof[0][k];
					v0[icell_face] = v_prof[0][k];
					w0[icell_face] = 0.0;         /// Perpendicular wind direction
				}
			}
   	}
  }

	// If number of sites are more than one
	// Apply 2D Barnes scheme to interpolate site velocity profiles to the whole domain
	else
	{
		rc_sum = 0.0;
		for (auto i = 0; i < num_sites; i++)
		{
			rc_val = 1000000.0;
			for (auto ii = 0; ii < num_sites; ii++)
			{
				xc = sensors[ii]->site_xcoord - sensors[i]->site_xcoord;
				yc = sensors[ii]->site_ycoord - sensors[i]->site_ycoord;
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
		for (auto j=0; j<ny; j++)
		{
			for (auto i=0; i<nx; i++)
			{
				sum_wm = 0.0;
				for (auto ii=0; ii<num_sites; ii++)
				{
					wm[ii][i][j] = exp((-1/lamda)*pow(sensors[ii]->site_xcoord-x[i],2.0)-(1/lamda)*pow(sensors[ii]->site_ycoord-y[j],2.0));
					wms[ii][i][j] = exp((-1/(s_gamma*lamda))*pow(sensors[ii]->site_xcoord-x[i],2.0)-(1/(s_gamma*lamda))*
										pow(sensors[ii]->site_ycoord-y[j],2.0));
					sum_wm += wm[ii][i][j];
				}
				if (sum_wm == 0)
				{
					for (auto ii = 0; ii<num_sites; ii++)
					{
						wm[ii][i][j] = 1e-20;
					}
				}
			}
		}

		for (auto k=1; k<nz; k++)
		{
			for (auto j=0; j<ny; j++)
			{
				for (auto i=0; i<nx; i++)
				{
					sum_wu = 0.0;
					sum_wv = 0.0;
					sum_wm = 0.0;
					for (auto ii=0; ii<num_sites; ii++)
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

			for (auto ii=0; ii<num_sites; ii++)
			{
				if(sensors[ii]->site_xcoord>0 && sensors[ii]->site_xcoord<(nx-1)*dx && sensors[ii]->site_ycoord>0 && sensors[ii]->site_ycoord<(ny-1)*dy)
				{
					for (auto j=0; j<ny; j++)
					{
						if (y[j]<sensors[ii]->site_ycoord)
						{
							jwork = j;
						}
					}
					for (auto i=0; i<nx; i++)
					{
						if (x[i]<sensors[ii]->site_xcoord)
						{
							iwork = i;
						}
					}
					dxx = sensors[ii]->site_xcoord-x[iwork];
					dyy = sensors[ii]->site_ycoord-y[jwork];
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

			for (auto j=0; j<ny; j++)
			{
				for (auto i=0; i<nx; i++)
				{
					sum_wu = 0.0;
					sum_wv = 0.0;
					sum_wm = 0.0;
					for (auto ii=0; ii<num_sites; ii++)
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
				//std::cout << "u0:   "  << u0[nx-1+j*nx+k*nx*ny] << std::endl;
			}
		}
	}

	if (z0_domain_flag == 1)
	{
		double sum_z0=0.0;
		double z0_effective;
		int height_id, blending_id, max_terrain_id;
		std::vector<float> blending_velocity, blending_theta;
		std::cout << "sum_z0:   " << sum_z0 << std::endl;
		for (auto i=0; i<nx; i++)
		{
			for (auto j=0; j<ny; j++)
			{
				id = i+j*nx;
				if (terrain_id[id] > max_terrain)
				{
					max_terrain = terrain_id[id];
					max_terrain_id = i+j*(nx-1);
				}
			}
		}

		for (auto i=0; i<nx; i++)
		{
			for (auto j=0; j<ny; j++)
			{
				id = i+j*nx;
				sum_z0 += log(z0_domain[id]+z[terrain_id[id]]);
			}
		}
		std::cout << "sum_z0:   " << sum_z0 << std::endl;
		z0_effective = exp(sum_z0/(nx*ny));
		std::cout << "z0_effective:   " << z0_effective << std::endl;
		std::cout << "blending_height:   " << blending_height << std::endl;
		std::cout << "average__one_overL:   " << average__one_overL << std::endl;
		std::cout << "max_terrain_id:   " << max_terrain_id << std::endl;
		std::cout << "terrain[max_terrain_id]:   " << terrain[max_terrain_id] << std::endl;
		blending_height = blending_height+terrain[max_terrain_id];
		for (auto k=0; k<z.size(); k++)
		{
			height_id = k+1;
			if (blending_height < z[k+1])
			{
				break;
			}
		}
		std::cout << "height_id:   " << height_id << "\t \t"<< z[height_id]<< std::endl;

		blending_velocity.resize( nx*ny, 0.0 );
		blending_theta.resize( nx*ny, 0.0 );

		for (auto i=0; i<nx; i++)
		{
			for (auto j=0; j<ny; j++)
			{
				blending_id = i+j*nx+height_id*nx*ny;
				id = i+j*nx;
				blending_velocity[id] = sqrt(pow(u0[blending_id],2.0)+pow(v0[blending_id],2.0));
				blending_theta[id] = atan2(v0[blending_id],u0[blending_id]);
				//std::cout << "id" << "\t \t"<< id  << "\t \t"<< "blending_velocity:   "  << "\t \t" <<blending_velocity[id] << std::endl;
			}
		}

		for (auto i=0; i<nx; i++)
		{
			for (auto j=0; j<ny; j++)
			{
				id = i+j*nx;
				if (blending_height*average__one_overL >= 0)
				{
					psi_first = 4.7*blending_height*average__one_overL;
				}
				else
				{
					x_temp = pow((1.0-15.0*blending_height*average__one_overL),0.25);
					psi_first = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
				}
				//std::cout << "psi:   "  << psi_first << std::endl;
				//std::cout << "terrain_id[id]:   " << terrain_id[id] << std::endl;
				//std::cout << "terrain_id[id]+height_id:   " << terrain_id[id]+height_id << std::endl;

				for (auto k = terrain_id[id]; k < height_id; k++)
				{
					if (z[k]*average__one_overL >= 0)
					{
						psi = 4.7*z[k]*average__one_overL;
					}
					else
					{
						x_temp = pow((1.0-15.0*z[k]*average__one_overL),0.25);
						psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
					}
					icell_face = i+j*nx+k*nx*ny;
					//std::cout << "psi:   "  << psi << std::endl;
					u_star = blending_velocity[id]*vk/(log((blending_height+z0_domain[id])/z0_domain[id])+psi_first);
					//std::cout << "ustar:   "  << u_star << std::endl;
					u0[icell_face] = (cos(blending_theta[id])*u_star/vk)*(log((z[k]+z0_domain[id])/z0_domain[id])+psi);
					v0[icell_face] = (sin(blending_theta[id])*u_star/vk)*(log((z[k]+z0_domain[id])/z0_domain[id])+psi);
					//std::cout << "i" << "\t \t"<< i <<  "\t \t"<<"j" << "\t \t"<< j <<  "\t \t"<<"k" << "\t \t"<< k  << "\t \t" << "u0[icell_face]:   "  << u0[icell_face] << std::endl;
				}

				for (auto k = height_id+1; k < nz-1; k++)
				{
					if (z[k]*average__one_overL >= 0)
					{
						psi = 4.7*z[k]*average__one_overL;
					}
					else
					{
						x_temp = pow((1.0-15.0*z[k]*average__one_overL),0.25);
						psi = -2.0*log(0.5*(1.0+x_temp))-log(0.5*(1.0+pow(x_temp,2.0)))+2.0*atan(x_temp)-0.5*M_PI;
					}
					//std::cout << "k" << "\t \t"<< k  << "\t \t" << "z:   "  << z[k] << std::endl;
					icell_face = i+j*nx+k*nx*ny;
					//std::cout << "blending_theta:   "  << blending_theta << std::endl;
					u_star = blending_velocity[id]*vk/(log((blending_height+z0_effective)/z0_effective)+psi_first);
					u0[icell_face] = (cos(blending_theta[id])*u_star/vk)*(log((z[k]+z0_effective)/z0_effective)+psi);
					v0[icell_face] = (sin(blending_theta[id])*u_star/vk)*(log((z[k]+z0_effective)/z0_effective)+psi);
					//std::cout << "k" << "\t \t"<< k  << "\t \t" << "u0[icell_face]:   "  << u0[icell_face] << std::endl;
				}

			}
		}


		/*ofstream outdata1;
		outdata1.open("Initial velocity.dat");
		if( !outdata1 ) {                 // File couldn't be opened
				cerr << "Error: file could not be opened" << endl;
				exit(1);
		}
		// Write data to file
		for (int k = 0; k < nz-1; k++){
				for (int j = 0; j < ny; j++){
						for (int i = 0; i < nx; i++){
							id = i+j*(nx-1);
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				int icell_face = i + j*nx + k*nx*ny;   /// Lineralized index for cell faced values
								outdata1 << "\t" << i << "\t" << j << "\t" << k << "\t \t"<< x[i] << "\t \t" << y[j] << "\t \t" << z[k]
						<< "\t \t"<< "\t \t" <<  u0[icell_face] <<"\t \t"<< "\t \t"<<v0[icell_face]
						<<"\t \t"<< "\t \t"<<w0[icell_face]<< "\t \t"<< "\t \t"<<terrain_id[id]<<endl;
						}
				}
		}
		outdata1.close();*/
	}

}



void Sensor::UTMConverter (float rlon, float rlat, float rx, float ry, int UTM_PROJECTION_ZONE, int iway)
{


/*

                S p e c f e m 3 D  V e r s i o n  2 . 1
                ---------------------------------------

           Main authors: Dimitri Komatitsch and Jeroen Tromp
     Princeton University, USA and CNRS / INRIA / University of Pau
  (c) Princeton University / California Institute of Technology and CNRS / INRIA / University of Pau
                              July 2012

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along
  with this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

*/

/*
  UTM (Universal Transverse Mercator) projection from the USGS
*/



/*
convert geodetic longitude and latitude to UTM, and back
use iway = ILONGLAT2UTM for long/lat to UTM, IUTM2LONGLAT for UTM to lat/long
a list of UTM zones of the world is available at www.dmap.co.uk/utmworld.htm
*/


/*
      CAMx v2.03

      UTM_GEO performs UTM to geodetic (long/lat) translation, and back.

      This is a Fortran version of the BASIC program "Transverse Mercator
      Conversion", Copyright 1986, Norman J. Berls (Stefan Musarra, 2/94)
      Based on algorithm taken from "Map Projections Used by the USGS"
      by John P. Snyder, Geological Survey Bulletin 1532, USDI.

      Input/Output arguments:

         rlon                  Longitude (deg, negative for West)
         rlat                  Latitude (deg)
         rx                    UTM easting (m)
         ry                    UTM northing (m)
         UTM_PROJECTION_ZONE  UTM zone
         iway                  Conversion type
                               ILONGLAT2UTM = geodetic to UTM
                               IUTM2LONGLAT = UTM to geodetic
*/


  int ILONGLAT2UTM = 0, IUTM2LONGLAT = 1;
  float PI = 3.141592653589793;
  float degrad = PI/180.0;
  float raddeg = 180.0/PI;
  float semimaj = 6378206.40;
  float semimin = 6356583.80;
  float scfa = 0.99960;

/*
  some extracts about UTM:

  There are 60 longitudinal projection zones numbered 1 to 60 starting at 180Â°W.
  Each of these zones is 6 degrees wide, apart from a few exceptions around Norway and Svalbard.
  There are 20 latitudinal zones spanning the latitudes 80Â°S to 84Â°N and denoted
  by the letters C to X, ommitting the letter O.
  Each of these is 8 degrees south-north, apart from zone X which is 12 degrees south-north.

  To change the UTM zone and the hemisphere in which the
  calculations are carried out, need to change the fortran code and recompile. The UTM zone is described
  actually by the central meridian of that zone, i.e. the longitude at the midpoint of the zone, 3 degrees
  from either zone boundary.
  To change hemisphere need to change the "north" variable:
  - north=0 for northern hemisphere and
  - north=10000000 (10000km) for southern hemisphere. values must be in metres i.e. north=10000000.

  Note that the UTM grids are actually Mercators which
  employ the standard UTM scale factor 0.9996 and set the
  Easting Origin to 500,000;
  the Northing origin in the southern
  hemisphere is kept at 0 rather than set to 10,000,000
  and this gives a uniform scale across the equator if the
  normal convention of selecting the Base Latitude (origin)
  at the equator (0 deg.) is followed.  Northings are
  positive in the northern hemisphere and negative in the
  southern hemisphere.
  */

  float north = 0.0;
  float east = 500000.0;

  float e2,e4,e6,ep2,xx,yy,dlat,dlon,zone,cm,cmr,delam;
  float f1,f2,f3,f4,rm,rn,t,c,a,e1,u,rlat1,dlat1,c1,t1,rn1,r1,d;
  float rx_save,ry_save,rlon_save,rlat_save;

  // save original parameters
  rlon_save = rlon;
  rlat_save = rlat;
  rx_save = rx;
  ry_save = ry;

  xx = 0.0;
  yy = 0.0;
  dlat = 0.0;
  dlon = 0.0;

  // define parameters of reference ellipsoid
  e2 = 1.0-pow((semimin/semimaj),2.0);
  e4 = pow(e2,2.0);
  e6 = e2*e4;
  ep2 = e2/(1.0-e2);

  if (iway == IUTM2LONGLAT)
  {
    xx = rx;
    yy = ry;
  }
  else
  {
    dlon = rlon;
    dlat = rlat;
  }

  // Set Zone parameters

  zone = UTM_PROJECTION_ZONE;
  // sets central meridian for this zone
  cm = zone*6.0 - 183.0;
  cmr = cm*degrad;

  // Lat/Lon to UTM conversion

  if (iway == ILONGLAT2UTM)
  {
    rlon = degrad*dlon;
    rlat = degrad*dlat;

    delam = dlon - cm;
    if (delam < -180.0)
    {
      delam = delam + 360.0;
    }
    if (delam > 180.0)
    {
      delam = delam - 360.0;
    }
    delam = delam*degrad;

    f1 = (1.0 - (e2/4.0) - 3.0*(e4/64.0) - 5.0*(e6/256))*rlat;
    f2 = 3.0*(e2/8.0) + 3.0*(e4/32.0) + 45.0*(e6/1024.0);
    f2 = f2*sin(2.0*rlat);
    f3 = 15.0*(e4/256.0)*45.0*(e6/1024.0);
    f3 = f3*sin(4.0*rlat);
    f4 = 35.0*(e6/3072.0);
    f4 = f4*sin(6.0*rlat);
    rm = semimaj*(f1 - f2 + f3 - f4);
    if (dlat == 90.0 || dlat == -90.0)
    {
      xx = 0.0;
      yy = scfa*rm;
    }
    else
    {
      rn = semimaj/sqrt(1.0 - e2*pow(sin(rlat),2.0));
      t = pow(tan(rlat),2.0);
      c = ep2*pow(cos(rlat),2.0);
      a = cos(rlat)*delam;

      f1 = (1.0 - t + c)*pow(a,3.0)/6.0;
      f2 = 5.0 - 18.0*t + pow(t,2.0) + 72.0*c - 58.0*ep2;
      f2 = f2*pow(a,5.0)/120.0;
      xx = scfa*rn*(a + f1 + f2);
      f1 = pow(a,2.0)/2.0;
      f2 = 5.0 - t + 9.0*c + 4.0*pow(c,2.0);
      f2 = f2*pow(a,4.0)/24.0;
      f3 = 61.0 - 58.0*t + pow(t,2.0) + 600.0*c - 330.0*ep2;
      f3 = f3*pow(a,6.0)/720.0;
      yy = scfa*(rm + rn*tan(rlat)*(f1 + f2 + f3));
    }
    xx = xx + east;
    yy = yy + north;
  }

  // UTM to Lat/Lon conversion

  else
  {
    xx = xx - east;
    yy = yy - north;
    e1 = sqrt(1.0 - e2);
    e1 = (1.0 - e1)/(1.0 + e1);
    rm = yy/scfa;
    u = 1.0 - (e2/4.0) - 3.0*(e4/64.0) - 5.0*(e6/256.0);
    u = rm/(semimaj*u);

    f1 = 3.0*(e1/2.0) - 27.0*pow(e1,3.0)/32.0;
    f1 = f1*sin(2.0*u);
    f2 = (21.0*pow(e1,2.0)/16.0) - 55.0*pow(e1,4.0)/32.0;
    f2 = f2*sin(4.0*u);
    f3 = 151.0*pow(e1,3.0)/96.0;
    f3 = f3*sin(6.0*u);
    rlat1 = u + f1 + f2 + f3;
    dlat1 = rlat1*raddeg;
    if (dlat1 >= 90.0 || dlat1 <= -90.0)
    {
      dlat1 = std::min(dlat1,90.0f) ;
      dlat1 = std::max(dlat1,-90.0f);
      dlon = cm;
    }
    else
    {
      c1 = ep2*pow(cos(rlat1),2.0);
      t1 = pow(tan(rlat1),2.0);
      f1 = 1.0 - e2*pow(sin(rlat1),2.0);
      rn1 = semimaj/sqrt(f1);
      r1 = semimaj*(1.0 - e2)/sqrt(pow(f1,3.0));
      d = xx/(rn1*scfa);

      f1 = rn1*tan(rlat1)/r1;
      f2 = pow(d,2.0)/2.0;
      f3 = 5.0*3.0*t1 + 10.0*c1 - 4.0*pow(c1,2.0) - 9.0*ep2;
      f3 = f3*pow(d,2.0)*pow(d,2.0)/24.0;
      f4 = 61.0 + 90.0*t1 + 298.0*c1 + 45.0*pow(t1,2.0) - 252.0*ep2 - 3.0*pow(c1,2.0);
      f4 = f4*pow(pow(d,2.0),3.0)/720.0;
      rlat = rlat1 - f1*(f2 - f3 + f4);
      dlat = rlat*raddeg;

      f1 = 1.0 + 2.0*t1 + c1;
      f1 = f1*pow(d,2.0)*d/6.0;
      f2 = 5.0 - 2.0*c1 + 28.0*t1 - 3.0*pow(c1,2.0) + 8.0*ep2 + 24.0*pow(t1,2.0);
      f2 = f2*pow(pow(d,2.0),2.0)*d/120.0;
      rlon = cmr + (d - f1 + f2)/cos(rlat1);
      dlon = rlon*raddeg;
      if (dlon < -180.0)
      {
        dlon = dlon + 360.0;
      }
      if (dlon > 180.0)
      {
        dlon = dlon - 360.0;
      }
    }
  }

  if (iway == IUTM2LONGLAT)
  {
    rlon = dlon;
    rlat = dlat;
    rx = rx_save;
    ry = ry_save;
  }
  else
  {
    rx = xx;
    ry = yy;
    rlon = rlon_save;
    rlat = rlat_save;
  }

}


void Sensor::getConvergence(float lon, float lat, int site_UTM_zone, float convergence)
{

	float temp_lon;
	temp_lon = (6.0*site_UTM_zone) - 183.0 -lon;
	convergence = atan(atan(temp_lon*M_PI/180.0)*asin(lat*M_PI/180.0))*(180.0/M_PI);

}
