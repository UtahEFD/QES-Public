#include "Canopy.h"



void Canopy::readCanopy(int nx, int ny, int nz, int landuse_flag, int num_canopies,int &lu_canopy_flag,
					std::vector<std::vector<std::vector<float>>> &canopy_atten,std::vector<std::vector<float>> &canopy_top)
{

	// This function needs to be updated when we can read land use data fom WRF or
	// other sources
	if (landuse_flag == 1)
	{
	}
	else
	{
		landuse_veg_flag=0;
		landuse_urb_flag=0;
		lu_canopy_flag=0;
	}

	if (lu_canopy_flag > 0)
	{
	}


}

// Function to apply the urban canopy parameterization
// Based on the version contain Lucas Ulmer's modifications
void Canopy::plantInitial(int nx, int ny, int nz, float vk, std::vector<int> &icellflag, std::vector<float> z, std::vector<double> &u0,
					std::vector<double> &v0, std::vector<std::vector<std::vector<float>>> &canopy_atten,
					std::vector<std::vector<float>> &canopy_top, std::vector<std::vector<float>> &canopy_top_index,
					std::vector<std::vector<float>> &canopy_ustar, std::vector<std::vector<float>> &canopy_z0,
					std::vector<std::vector<float>> &canopy_d)
{

	float u_H;                  /**< velocity at the height of the canopy */
	float avg_atten;						/**< average attenuation of the canopy */
	float veg_vel_frac;					/**< vegetation velocity fraction */
	int num_atten;

	// Call regression to define ustar and and surface roughness of the canopy
	regression(nx,ny,nz,vk,z.data(),u0.data(),v0.data(),canopy_atten,canopy_top,canopy_top_index,canopy_ustar,canopy_z0);

	for (int j=0; j<ny-1; j++)
	{
		for (int i=0; i<nx-1; i++)
		{
			if (canopy_top[i][j] > 0)
			{
				// Call the bisection method to find the root
				canopy_d[i][j] = bisection(canopy_ustar[i][j],canopy_z0[i][j], canopy_top[i][j],
								canopy_atten[i][j][canopy_top_index[i][j]],vk,0.0);
				std::cout << "d passed from bisection is:" << canopy_d[i][j] << "\n";
				if (canopy_d[i][j] == 10000)
				{
					std::cout << "bisection failed to converge" << "\n";
					canopy_d[i][j] = canopy_slope_match(canopy_z0[i][j], canopy_top[i][j],
									canopy_atten[i][j][canopy_top_index[i][j]]);
				}
				u_H = (canopy_ustar[i][j]/vk)*log((canopy_top[i][j]-canopy_d[i][j])/canopy_z0[i][j]);
				for (int k=1; k < nz; k++)
				{
					if (z[k] < canopy_top[i][j])
					{
						if (canopy_atten[i][j][k] > 0)
						{
							avg_atten = canopy_atten[i][j][k];

							if (canopy_atten[i][j][k+1]!=canopy_atten[i][j][k] ||
											canopy_atten[i][j][k-1]!=canopy_atten[i][j][k])
							{
								num_atten = 1;
								if (canopy_atten[i][j][k+1] > 0)
								{
									avg_atten += canopy_atten[i][j][k+1];
									num_atten += 1;
								}
								if (canopy_atten[i][j][k-1] > 0)
								{
									avg_atten += canopy_atten[i][j][k-1];
									num_atten += 1;
								}
								avg_atten /= num_atten;
							}
							veg_vel_frac = log((canopy_top[i][j]-canopy_d[i][j])/canopy_z0[i][j])*exp(avg_atten*
											((z[k]/canopy_top[i][j])-1))/log(z[k]/canopy_z0[i][j]);
							if (veg_vel_frac > 1 || veg_vel_frac < 0)
							{
								veg_vel_frac = 1;
							}
							int icell_face = i + j*nx + k*nx*ny;
							u0[icell_face] *= veg_vel_frac;
							v0[icell_face] *= veg_vel_frac;
							if (j < ny-2)
							{
								if (canopy_atten[i][j+1][k] == 0)
								{
									v0[icell_face+nx] *= veg_vel_frac;
								}
							}
							if (i < nx-2)
							{
								if(canopy_atten[i+1][j][k] == 0)
								{
									u0[icell_face+1] *= veg_vel_frac;
								}
							}
							int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
							if( icellflag[icell_cent] > 0)
							{
								icellflag[icell_cent] = 8;
							}
						}
					}
					else
					{
						veg_vel_frac = log((z[k]-canopy_d[i][j])/canopy_z0[i][j])/log(z[k]/canopy_z0[i][j]);
						if (veg_vel_frac > 1 || veg_vel_frac < 0)
						{
							veg_vel_frac = 1;
						}
						int icell_face = i + j*nx + k*nx*ny;
						u0[icell_face] *= veg_vel_frac;
						v0[icell_face] *= veg_vel_frac;
						if (j < ny-2)
						{
							if(canopy_atten[i][j+1][canopy_top_index[i][j]] == 0)
							{
								v0[icell_face+nx] *= veg_vel_frac;
							}
						}
						if (i < nx-2)
						{
							if (canopy_atten[i+1][j][canopy_top_index[i][j]] == 0)
							{
								u0[icell_face+1] *= veg_vel_frac;
							}
						}
					}
				}
			}
		}
	}

}


void Canopy::regression(int nx, int ny, int nz, float vk, float* z, double *u0, double *v0,
					std::vector<std::vector<std::vector<float>>> &canopy_atten, std::vector<std::vector<float>> &canopy_top,
					std::vector<std::vector<float>> &canopy_top_index, std::vector<std::vector<float>> &canopy_ustar,
					std::vector<std::vector<float>> &canopy_z0)
{

	int k_top, counter;
	float sum_x, sum_y, sum_xy, sum_x_sq, local_mag;
	float y, xm, ym;

	for (int i=0; i<nx-1; i++)
	{
		for (int j=0; j<ny-1; j++)
		{
			if (canopy_top[i][j] > 0)                      // If the cell is inside a vegetation element
			{
				for (int k=1; k<nz-2; k++)
				{
					canopy_top_index[i][j] = k;
					if (canopy_top[i][j] < z[k+1])
						break;
				}
				for (int k=canopy_top_index[i][j]; k<nz-2; k++)
				{
					k_top = k;
					if (2*canopy_top[i][j] < z[k+1])
						break;
				}
				if (k_top == canopy_top_index[i][j])
					k_top = canopy_top_index[i][j]+1;
				if (k_top > nz-1)
					k_top = nz-1;
				sum_x = 0;
				sum_y = 0;
				sum_xy = 0;
				sum_x_sq = 0;
				counter = 0;
				for (int k=canopy_top_index[i][j]; k<=k_top; k++)
				{
					counter +=1;
					int icell_face = i + j*nx + k*nx*ny;
					local_mag = sqrt(pow(u0[icell_face],2.0)+pow(v0[icell_face],2.0));
					y = log(z[k]);
					sum_x += local_mag;
					sum_y += y;
					sum_xy += local_mag*y;
					sum_x_sq += pow(local_mag,2.0);
				}
				canopy_ustar[i][j] = vk*(((counter*sum_x_sq)-pow(sum_x,2.0))/((counter*sum_xy)-(sum_x*sum_y)));
				xm = sum_x/counter;
				ym = sum_y/counter;
				canopy_z0[i][j] = exp(ym-((vk/canopy_ustar[i][j]))*xm);
			}
		}
	}

}




float Canopy::bisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m)
{

	int iter;
	float tol, uhc, d, d1, d2, fi, fnew;

	tol = z0/100;
	fnew = tol*10;

	d1 = z0;
	d2 = canopy_top;
	d = (d1+d2)/2;

	uhc = (ustar/vk)*(log((canopy_top-d1)/z0)+psi_m);
	fi = ((canopy_atten*uhc*vk)/ustar)-canopy_top/(canopy_top-d1);

	if (canopy_atten > 0)
	{
		iter = 0;
		while (iter < 200 && abs(fnew) > tol && d < canopy_top && d > z0)
		{
			iter += 1;
			d = (d1+d2)/2;
			uhc = (ustar/vk)*(log((canopy_top-d)/z0)+psi_m);
			fnew = ((canopy_atten*uhc*vk)/ustar) - canopy_top/(canopy_top-d);
			if(fnew*fi>0)
			{
				d1 = d;
			}
			else if(fnew*fi<0)
			{
				d2 = d;
			}
		}
		if (d > canopy_top)
		{
			d = 10000;
		}
	}
	else
	{
		d = 0.99*canopy_top;
	}

	return d;

}


float Canopy::canopy_slope_match(float z0, float canopy_top, float canopy_atten)
{

	int iter;
	float tol, d, d1, d2, f;

	tol = z0/100;
	f = tol*10;

	if (z0 < canopy_top)
	{
		d1 = z0;
	}
	else if (z0 > canopy_top)
	{
		d1 = 0.1;
	}
	d2 = canopy_top;
	d = (d1+d2)/2;

	if (canopy_atten > 0)
	{
		iter = 0;
		while (iter < 200 && abs(f) > tol && d < canopy_top && d > z0)
		{
			iter += 1;
			d = (d1+d2)/2;
			f = log ((canopy_top-d)/z0) - (canopy_top/(canopy_atten*(canopy_top-d)));
			if(f > 0)
			{
				d1 = d;
			}
			else if(f<0)
			{
				d2 = d;
			}
		}
		if (d > canopy_top)
		{
			d = 0.7*canopy_top;
		}
	}
	else
	{
		d = 10000;
	}

	return d;

}
