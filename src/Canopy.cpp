#include "Canopy.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

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

void Canopy::defineCanopy(WINDSGeneralData* WGD)
{

    float ray_intersect;
    unsigned int num_crossing, vert_id, start_poly;

    // Define start index of the canopy in z-direction
    for (auto k=1; k<WGD->z.size(); k++)
    {
        k_start = k;
        if (base_height <= WGD->z[k])
        {
            break;
        }
    }

    // Define end index of the canopy in z-direction   Note that 0u
    // means 0 unsigned
    for (auto k=0; k<WGD->z.size(); k++)
    {
        k_end = k+1;
        if (base_height+H < WGD->z[k+1])
        {
            break;
        }
    }

    // Loop to calculate maximum and minimum of x and y values of the building
    x_min = x_max = polygonVertices[0].x_poly;
    y_min = x_max = polygonVertices[0].y_poly;
    for (auto id=1; id<polygonVertices.size(); id++)
    {
        if (polygonVertices[id].x_poly > x_max)
        {
            x_max = polygonVertices[id].x_poly;
        }
        if (polygonVertices[id].x_poly < x_min)
        {
            x_min = polygonVertices[id].x_poly;
        }
        if (polygonVertices[id].y_poly > y_max)
        {
            y_max = polygonVertices[id].y_poly;
        }
        if (polygonVertices[id].y_poly < y_min)
        {
            y_min = polygonVertices[id].y_poly;
        }
    }

    i_start = (x_min/WGD->dx);       // Index of canopy start location in x-direction
    i_end = (x_max/WGD->dx)+1;       // Index of canopy end location in x-direction
    j_start = (y_min/WGD->dy);       // Index of canopy end location in y-direction
    j_end = (y_max/WGD->dy)+1;       // Index of canopy start location in y-direction

    // Find out which cells are going to be inside the polygone
    // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
    // Check the center of each cell, if it's inside, set that cell to building
    for (auto j=j_start; j<j_end; j++)
    {
        y_cent = (j+0.5)*WGD->dy;         // Center of cell y coordinate
        for (auto i=i_start; i<i_end; i++)
        {
            x_cent = (i+0.5)*WGD->dx;
            vert_id = 0;               // Node index
            start_poly = vert_id;
            num_crossing = 0;
            while (vert_id < polygonVertices.size()-1)
            {
                if ( (polygonVertices[vert_id].y_poly<=y_cent && polygonVertices[vert_id+1].y_poly>y_cent) ||
                     (polygonVertices[vert_id].y_poly>y_cent && polygonVertices[vert_id+1].y_poly<=y_cent) )
                {
                    ray_intersect = (y_cent-polygonVertices[vert_id].y_poly)/(polygonVertices[vert_id+1].y_poly-polygonVertices[vert_id].y_poly);
                    if (x_cent < (polygonVertices[vert_id].x_poly+ray_intersect*(polygonVertices[vert_id+1].x_poly-polygonVertices[vert_id].x_poly)))
                    {
                        num_crossing += 1;
                    }
                }
                vert_id += 1;
                if (polygonVertices[vert_id].x_poly == polygonVertices[start_poly].x_poly &&
                    polygonVertices[vert_id].y_poly == polygonVertices[start_poly].y_poly)
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
                    int icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
                    if( WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)
                    {
                        WGD->icellflag[icell_cent] = 11;           // Canopy cell
                    }
                }
            }

        }
    }

    for (auto j=j_start; j<j_end-1; j++)
    {
        for (auto i=i_start; i<i_end-1; i++)
        {
            for (auto k=k_start; k<k_end; k++)
            {
                int icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
                if (WGD->icellflag[icell_cent] == 11)       // if the cell is defined as canopy
                {
                    int id = i+j*(WGD->nx-1);
                    WGD->canopy_top[id] = base_height+H;
                    WGD->canopy_atten[icell_cent] = atten;     // initiate all attenuation coefficients to the canopy coefficient
                }
            }
        }
    }
}


// Function to apply the urban canopy parameterization
// Based on the version contain Lucas Ulmer's modifications
void Canopy::plantInitial(WINDSGeneralData* WGD)
{

    float avg_atten;						/**< average attenuation of the canopy */
    float veg_vel_frac;					/**< vegetation velocity fraction */
    int num_atten;

    // Call regression to define ustar and surface roughness of the canopy
    regression(WGD);

    for (auto j=j_start; j<j_end-1; j++)
    {
        for (auto i=i_start; i<i_end-1; i++)
        {
            int id = i+j*(WGD->nx-1);
            if (WGD->canopy_top[id] > 0)
            {
                // Call the bisection method to find the root
                int icell_cent = i+j*(WGD->nx-1)+WGD->canopy_top_index[id]*(WGD->nx-1)*(WGD->ny-1);
                WGD->canopy_d[id] = WGD->canopyBisection(WGD->canopy_ustar[id],WGD->canopy_z0[id], WGD->canopy_top[id],
                                                         WGD->canopy_atten[icell_cent],WGD->vk,0.0);

                if (WGD->canopy_d[id] == 10000)
                {
                    std::cout << "bisection failed to converge" << "\n";
                    WGD->canopy_d[id] = canopy_slope_match(WGD->canopy_z0[id], WGD->canopy_top[id], WGD->canopy_atten[icell_cent]);
                }

                for (auto k=1; k < WGD->nz-1; k++)
                {
                    if (WGD->z[k] < WGD->canopy_top[id])
                    {
                        if (WGD->canopy_atten[icell_cent] > 0)
                        {
                            icell_cent = i+j*(WGD->nx-1)+k*(WGD->nx-1)*(WGD->ny-1);
                            avg_atten = WGD->canopy_atten[icell_cent];

                            if (WGD->canopy_atten[icell_cent+(WGD->nx-1)*(WGD->ny-1)]!=WGD->canopy_atten[icell_cent] ||
                                WGD->canopy_atten[icell_cent-(WGD->nx-1)*(WGD->ny-1)]!=WGD->canopy_atten[icell_cent])
                            {
                                num_atten = 1;
                                if (WGD->canopy_atten[icell_cent+(WGD->nx-1)*(WGD->ny-1)] > 0)
                                {
                                    avg_atten += WGD->canopy_atten[icell_cent+(WGD->nx-1)*(WGD->ny-1)];
                                    num_atten += 1;
                                }
                                if (WGD->canopy_atten[icell_cent-(WGD->nx-1)*(WGD->ny-1)] > 0)
                                {
                                    avg_atten += WGD->canopy_atten[icell_cent-(WGD->nx-1)*(WGD->ny-1)];
                                    num_atten += 1;
                                }
                                avg_atten /= num_atten;
                            }
                            veg_vel_frac = log((WGD->canopy_top[id] - WGD->canopy_d[id])/WGD->canopy_z0[id])*exp(avg_atten*
                                                                                                               ((WGD->z[k]/WGD->canopy_top[id])-1))/log(WGD->z[k]/WGD->canopy_z0[id]);
                            if (veg_vel_frac > 1 || veg_vel_frac < 0)
                            {
                                veg_vel_frac = 1;
                            }
                            int icell_face = i + j*WGD->nx + k*WGD->nx*WGD->ny;
                            WGD->u0[icell_face] *= veg_vel_frac;
                            WGD->v0[icell_face] *= veg_vel_frac;
                            if (j < WGD->ny-2)
                            {
                                if (WGD->canopy_atten[icell_cent+(WGD->nx-1)] == 0)
                                {
                                    WGD->v0[icell_face+WGD->nx] *= veg_vel_frac;
                                }
                            }
                            if (i < WGD->nx-2)
                            {
                                if(WGD->canopy_atten[icell_cent+1] == 0)
                                {
                                    WGD->u0[icell_face+1] *= veg_vel_frac;
                                }
                            }
                        }
                    }
                    else
                    {
                        veg_vel_frac = log((WGD->z[k]-WGD->canopy_d[id])/WGD->canopy_z0[id])/log(WGD->z[k]/WGD->canopy_z0[id]);
                        if (veg_vel_frac > 1 || veg_vel_frac < 0)
                        {
                            veg_vel_frac = 1;
                        }
                        int icell_face = i + j*WGD->nx + k*WGD->nx*WGD->ny;
                        WGD->u0[icell_face] *= veg_vel_frac;
                        WGD->v0[icell_face] *= veg_vel_frac;
                        if (j < WGD->ny-2)
                        {
                            icell_cent = i+j*(WGD->nx-1)+WGD->canopy_top_index[id]*(WGD->nx-1)*(WGD->ny-1);
                            if(WGD->canopy_atten[icell_cent +(WGD->nx-1)] == 0)
                            {
                                WGD->v0[icell_face+WGD->nx] *= veg_vel_frac;
                            }
                        }
                        if (i < WGD->nx-2)
                        {
                            if (WGD->canopy_atten[icell_cent+1] == 0)
                            {
                                WGD->u0[icell_face+1] *= veg_vel_frac;
                            }
                        }
                    }
                }
            }
        }
    }

}


void Canopy::regression(WINDSGeneralData* WGD)
{

    int k_top, counter;
    float sum_x, sum_y, sum_xy, sum_x_sq, local_mag;
    float y, xm, ym;

    for (auto j=j_start; j<j_end-1; j++)
    {
        for (auto i=i_start; i<i_end-1; i++)
        {
            int id = i+j*(WGD->nx-1);
            if (WGD->canopy_top[id] > 0)
            {
                for (auto k=1; k<WGD->nz-2; k++)
                {
                    WGD->canopy_top_index[id] = k;
                    if (WGD->canopy_top[id] < WGD->z[k+1])
                        break;
                }
                for (auto k=WGD->canopy_top_index[id]; k<WGD->nz-2; k++)
                {
                    k_top = k;
                    if (2*WGD->canopy_top[id] < WGD->z[k+1])
                        break;
                }
                if (k_top == WGD->canopy_top_index[id])
                {
                    k_top = WGD->canopy_top_index[id]+1;
                }
                if (k_top > WGD->nz-1)
                {
                    k_top = WGD->nz-1;
                }
                sum_x = 0;
                sum_y = 0;
                sum_xy = 0;
                sum_x_sq = 0;
                counter = 0;
                for (auto k=WGD->canopy_top_index[id]; k<=k_top; k++)
                {
                    counter +=1;
                    int icell_face = i + j*WGD->nx + k*WGD->nx*WGD->ny;
                    local_mag = sqrt(pow(WGD->u0[icell_face],2.0)+pow(WGD->v0[icell_face],2.0));
                    y = log(WGD->z[k]);
                    sum_x += local_mag;
                    sum_y += y;
                    sum_xy += local_mag*y;
                    sum_x_sq += pow(local_mag,2.0);
                }
                WGD->canopy_ustar[id] = WGD->vk*(((counter*sum_x_sq)-pow(sum_x,2.0))/((counter*sum_xy)-(sum_x*sum_y)));
                xm = sum_x/counter;
                ym = sum_y/counter;
                WGD->canopy_z0[id] = exp(ym-((WGD->vk/WGD->canopy_ustar[id]))*xm);
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


void Canopy::canopyVegetation(WINDSGeneralData* WGD)
{

    // When THIS canopy calls this function, we need to do the
    // following:
    //readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
    //canopy_atten, canopy_top);

    // here because the array that holds this all Building*
    defineCanopy(WGD);

    plantInitial(WGD);		// Apply canopy parameterization
}
