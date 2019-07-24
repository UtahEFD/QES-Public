#include "Canopy.h"



/*void Canopy::readCanopy(int nx, int ny, int nz, int landuse_flag, int num_canopies,int &lu_canopy_flag,
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


}*/

void Canopy::defineCanopy(URBGeneralData* UGD)
{

	float ray_intersect;
	int num_crossing, vert_id, start_poly;

	// Define start index of the canopy in z-direction
	for (auto k=1; k<UGD->z.size(); k++)
	{
		k_start = k;
		if (base_height <= UGD->z[k])
		{
			break;
		}
	}

	// Define end index of the canopy in z-direction
	for (auto k=0; k<UGD->z.size(); k++)
	{
		k_end = k+1;
		if (base_height+H < UGD->z[k+1])
		{
			break;
		}
	}

	// Loop to calculate maximum and minimum of x and y values of the building
	x_min = x_max = polygonVortices[0].x_poly;
	y_min = x_max = polygonVortices[0].y_poly;
	for (auto id=1; id<polygonVortices.size(); id++)
	{
		if (polygonVortices[id].x_poly > x_max)
		{
			x_max = polygonVortices[id].x_poly;
		}
		if (polygonVortices[id].x_poly < x_min)
		{
			x_min = polygonVortices[id].x_poly;
		}
		if (polygonVortices[id].y_poly > y_max)
		{
			y_max = polygonVortices[id].y_poly;
		}
		if (polygonVortices[id].y_poly < y_min)
		{
			y_min = polygonVortices[id].y_poly;
		}
	}

	i_start = (x_min/UGD->dx);       // Index of canopy start location in x-direction
	i_end = (x_max/UGD->dx)+1;       // Index of canopy end location in x-direction
	j_start = (y_min/UGD->dy);       // Index of canopy end location in y-direction
	j_end = (y_max/UGD->dy)+1;       // Index of canopy start location in y-direction

	// Find out which cells are going to be inside the polygone
	// Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
	// Check the center of each cell, if it's inside, set that cell to building
	for (auto j=j_start; j<j_end; j++)
	{
		y_cent = (j+0.5)*UGD->dy;         // Center of cell y coordinate
		for (auto i=i_start; i<i_end; i++)
		{
			x_cent = (i+0.5)*UGD->dx;
			vert_id = 0;               // Node index
			start_poly = vert_id;
			num_crossing = 0;
			while (vert_id < polygonVortices.size()-1)
			{
				if ( (polygonVortices[vert_id].y_poly<=y_cent && polygonVortices[vert_id+1].y_poly>y_cent) ||
						 (polygonVortices[vert_id].y_poly>y_cent && polygonVortices[vert_id+1].y_poly<=y_cent) )
				{
					ray_intersect = (y_cent-polygonVortices[vert_id].y_poly)/(polygonVortices[vert_id+1].y_poly-polygonVortices[vert_id].y_poly);
					if (x_cent < (polygonVortices[vert_id].x_poly+ray_intersect*(polygonVortices[vert_id+1].x_poly-polygonVortices[vert_id].x_poly)))
					{
						num_crossing += 1;
					}
				}
				vert_id += 1;
				if (polygonVortices[vert_id].x_poly == polygonVortices[start_poly].x_poly &&
						polygonVortices[vert_id].y_poly == polygonVortices[start_poly].y_poly)
				{
					vert_id += 1;
					start_poly = vert_id;
				}
			}
			// if num_crossing is odd = cell is oustside of the polygon
			// if num_crossing is even = cell is inside of the polygon
			if ( (num_crossing%2) != 0 )
			{
				for (auto k=k_start; k<k_end; k++)
				{
					int icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
					if( UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
					{
						UGD->icellflag[icell_cent] = 6;           // Canopy cell
					}
				}
			}

		}
	}

	for (auto j=j_start; j<j_end; j++)
	{
		for (auto i=i_start; i<i_end; i++)
		{
			for (auto k=k_start; k<k_end; k++)
			{
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
				if (icellflag[icell_cent] == 6)       // if the cell is defined as canopy
				{
					int id = i+j*(UGD->nx-1);
					UGD->canopy_top[id] = base_height+H;
					UGD->canopy_atten[icell_cent] = atten;     // initiate all attenuation coefficients to the canopy coefficient
				}
			}
		}
	}
}


// Function to apply the urban canopy parameterization
// Based on the version contain Lucas Ulmer's modifications
void Canopy::plantInitial(URBGeneralData* UGD)
{

	float u_H;                  /**< velocity at the height of the canopy */
	float avg_atten;						/**< average attenuation of the canopy */
	float veg_vel_frac;					/**< vegetation velocity fraction */
	int num_atten;

	// Call regression to define ustar and surface roughness of the canopy
	regression(UGD);

	for (auto j=j_start; j<j_end; j++)
	{
		for (auto i=i_start; i<i_end; i++)
		{
			int id = i+j*(UGD->nx-1);
			if (UGD->canopy_top[id] > 0)
			{
				// Call the bisection method to find the root
				int icell_cent = i+j*(UGD->nx-1)+UGD->canopy_top_index[id]*(UGD->nx-1)*(UGD->ny-1);
				UGD->canopy_d[id] = UGD->canopyBisection(UGD->canopy_ustar[id],UGD->canopy_z0[id], UGD->canopy_top[id],
								UGD->canopy_atten[icell_cent],UGD->vk,0.0);
				if (UGD->canopy_d[id] == 10000)
				{
					std::cout << "bisection failed to converge" << "\n";
					UGD->canopy_d[id] = canopy_slope_match(UGD->canopy_z0[id], UGD->canopy_top[id], UGD->canopy_atten[icell_cent]);
				}
				u_H = (UGD->canopy_ustar[id]/UGD->vk)*log((UGD->canopy_top[id]-UGD->canopy_d[id])/UGD->canopy_z0[id]);
				for (auto k=1; k < UGD->nz; k++)
				{
					if (UGD->z[k] < UGD->canopy_top[id])
					{
						if (UGD->canopy_atten[icell_cent] > 0)
						{
							icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
							avg_atten = UGD->canopy_atten[icell_cent];

							if (UGD->canopy_atten[icell_cent+(UGD->nx-1)*(UGD->ny-1)]!=UGD->canopy_atten[icell_cent] ||
									UGD->canopy_atten[icell_cent-(UGD->nx-1)*(UGD->ny-1)]!=UGD->canopy_atten[icell_cent])
							{
								num_atten = 1;
								if (UGD->canopy_atten[icell_cent+(UGD->nx-1)*(UGD->ny-1)] > 0)
								{
									avg_atten += UGD->canopy_atten[icell_cent+(UGD->nx-1)*(UGD->ny-1)];
									num_atten += 1;
								}
								if (UGD->canopy_atten[icell_cent-(UGD->nx-1)*(UGD->ny-1)] > 0)
								{
									avg_atten += UGD->canopy_atten[icell_cent-(UGD->nx-1)*(UGD->ny-1)];
									num_atten += 1;
								}
								avg_atten /= num_atten;
							}
							veg_vel_frac = log((ufd->canopy_top[id]-UGD->canopy_d[id])/UGD->canopy_z0[id])*exp(avg_atten*
											((UGD->z[k]/UGD->canopy_top[id])-1))/log(UGD->z[k]/UGD->canopy_z0[id]);
							if (veg_vel_frac > 1 || veg_vel_frac < 0)
							{
								veg_vel_frac = 1;
							}
							int icell_face = i + j*UGD->nx + k*UGD->nx*UGD->ny;
							UGD->u0[icell_face] *= veg_vel_frac;
							UGD->v0[icell_face] *= veg_vel_frac;
							if (j < UGD->ny-2)
							{
								if (UGD->canopy_atten[icell_cent+(UGD->nx-1)] == 0)
								{
									UGD->v0[icell_face+UGD->nx] *= veg_vel_frac;
								}
							}
							if (i < UGD->nx-2)
							{
								if(UGD->canopy_atten[icell_cent+1] == 0)
								{
									UGD->u0[icell_face+1] *= veg_vel_frac;
								}
							}
						}
					}
					else
					{
						veg_vel_frac = log((UGD->z[k]-UGD->canopy_d[id])/UGD->canopy_z0[id])/log(UGD->z[k]/UGD->canopy_z0[id]);
						if (veg_vel_frac > 1 || veg_vel_frac < 0)
						{
							veg_vel_frac = 1;
						}
						int icell_face = i + j*UGD->nx + k*UGD->nx*UGD->ny;
						UGD->u0[icell_face] *= veg_vel_frac;
						UGD->v0[icell_face] *= veg_vel_frac;
						if (j < UGD->ny-2)
						{
							icell_cent = i+j*(UGD->nx-1)+UGD->canopy_top_index[id]*(UGD->nx-1)*(UGD->ny-1);
							if(UGD->canopy_atten[icell_cent +(UGD->nx-1)] == 0)
							{
								UGD->v0[icell_face+UGD->nx] *= veg_vel_frac;
							}
						}
						if (i < UGD->nx-2)
						{
							if (UGD->canopy_atten[icell_cent+1] == 0)
							{
								UGD->u0[icell_face+1] *= veg_vel_frac;
							}
						}
					}
				}
			}
		}
	}

}


void Canopy::regression(URBGeneralData* UGD)
{

	int k_top, counter;
	float sum_x, sum_y, sum_xy, sum_x_sq, local_mag;
	float y, xm, ym;

	for (auto j=j_start; j<j_end; j++)
	{
		for (auto i=i_start; i<i_end; i++)
		{
			int id = i+j*(UGD->nx-1);
			if (UGD->canopy_top[id] > 0)
			{
				for (auto k=1; k<UGD->nz-2; k++)
				{
					UGD->canopy_top_index[id] = k;
					if (UGD->canopy_top[id] < UGD->z[k+1])
						break;
				}
				for (auto k=UGD->canopy_top_index[id]; k<UGD->nz-2; k++)
				{
					k_top = k;
					if (2*UGD->canopy_top[id] < UGD->z[k+1])
						break;
				}
				if (k_top == UGD->canopy_top_index[id])
				{
					k_top = UGD->canopy_top_index[id]+1;
				}
				if (k_top > UGD->nz-1)
				{
					k_top = UGD->nz-1;
				}
				sum_x = 0;
				sum_y = 0;
				sum_xy = 0;
				sum_x_sq = 0;
				counter = 0;
				for (auto k=UGD->canopy_top_index[id]; k<=k_top; k++)
				{
					counter +=1;
					int icell_face = i + j*UGD->nx + k*UGD->nx*UGD->ny;
					local_mag = sqrt(pow(UGD->u0[icell_face],2.0)+pow(UGD->v0[icell_face],2.0));
					y = log(UGD->z[k]);
					sum_x += local_mag;
					sum_y += y;
					sum_xy += local_mag*y;
					sum_x_sq += pow(local_mag,2.0);
				}
				UGD->canopy_ustar[id] = UGD->vk*(((counter*sum_x_sq)-pow(sum_x,2.0))/((counter*sum_xy)-(sum_x*sum_y)));
				xm = sum_x/counter;
				ym = sum_y/counter;
				UGD->canopy_z0[id] = exp(ym-((UGD->vk/UGD->canopy_ustar[id]))*xm);
			}
		}
	}

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


void Canopy::canopyVegetation(URBGeneralData* UGD)
{

    // When THIS canopy calls this function, we need to do the
    // following:
    //readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
               //canopy_atten, canopy_top);

    // here because the array that holds this all Building*
    defineCanopy(UGD);

    plantInitial(UGD);		// Apply canopy parameterization
}
