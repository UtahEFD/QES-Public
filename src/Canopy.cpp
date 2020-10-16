#include "Canopy.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

void Canopy::canopyDefineBoundary(WINDSGeneralData* WGD,int cellFlagToUse)
{

    float ray_intersect;
    unsigned int num_crossing, vert_id, start_poly;

    // Define start index of the canopy in z-direction
    for (auto k=1; k<WGD->z.size(); k++) {
        k_start = k;
        if (base_height <= WGD->z[k]) {
            break;
        }
    }
  
    // Define end index of the canopy in z-direction   
    // Note that 0u means 0 unsigned
    for (auto k=0; k<WGD->z.size(); k++) {
        k_end = k+1;
        if (base_height+H < WGD->z[k+1]) {
            break;
        }
    }

    // Loop to calculate maximum and minimum of x and y values of the building
    x_min = x_max = polygonVertices[0].x_poly;
    y_min = x_max = polygonVertices[0].y_poly;
    for (size_t id=1; id<polygonVertices.size(); id++) {
        if (polygonVertices[id].x_poly > x_max) {
            x_max = polygonVertices[id].x_poly;
        }
        if (polygonVertices[id].x_poly < x_min) {
            x_min = polygonVertices[id].x_poly;
        }
        if (polygonVertices[id].y_poly > y_max) {
            y_max = polygonVertices[id].y_poly;
        }
        if (polygonVertices[id].y_poly < y_min) {
            y_min = polygonVertices[id].y_poly;
        }
    }
  
    // i_start and i_end are faces and not cells
    i_start = x_min/WGD->dx;       // Index of canopy start location in x-direction
    i_end = x_max/WGD->dx+1;       // Index of canopy end location in x-direction
    // j_start and j_end are faces and not cells
    j_start = y_min/WGD->dy;       // Index of canopy end location in y-direction
    j_end = y_max/WGD->dy+1;       // Index of canopy start location in y-direction
    
    // size of the canopy array -> with ghost cell before and after (hence +2) 
    nx_canopy = (i_end-i_start-1)+2;
    ny_canopy = (j_end-j_start-1)+2;
    
    // number of cell cell-center elements (2D) 
    numcell_cent_2d = nx_canopy*ny_canopy;
    
    // Resize the canopy-related vectors
    canopy_bot.resize( numcell_cent_2d , 0.0 );    
    canopy_top.resize( numcell_cent_2d , 0.0 );

    canopy_bot_index.resize( numcell_cent_2d , 0 );
    canopy_top_index.resize( numcell_cent_2d , -1 );
    
    canopy_base.resize( numcell_cent_2d , 0.0 );
    canopy_height.resize( numcell_cent_2d , 0.0 );

    canopy_z0.resize( numcell_cent_2d, 0.0 );
    canopy_ustar.resize( numcell_cent_2d, 0.0 );
    canopy_d.resize( numcell_cent_2d, 0.0 );

    // Find out which cells are going to be inside the polygone
    // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
    // Check the center of each cell, if it's inside, set that cell to building
    for (auto j=j_start; j<j_end; j++) {
        // Center of cell y coordinate
        y_cent = (j+0.5)*WGD->dy;         
        for (auto i=i_start; i<i_end; i++) {
            x_cent = (i+0.5)*WGD->dx;
            // Node index
            vert_id = 0;               
            start_poly = vert_id;
            num_crossing = 0;
            while (vert_id < polygonVertices.size()-1) {
                if ( (polygonVertices[vert_id].y_poly<=y_cent && polygonVertices[vert_id+1].y_poly>y_cent) ||
                     (polygonVertices[vert_id].y_poly>y_cent && polygonVertices[vert_id+1].y_poly<=y_cent) ) {
                    ray_intersect = (y_cent-polygonVertices[vert_id].y_poly)/
                        (polygonVertices[vert_id+1].y_poly-polygonVertices[vert_id].y_poly);
                    if (x_cent < (polygonVertices[vert_id].x_poly+ray_intersect*
                                  (polygonVertices[vert_id+1].x_poly-polygonVertices[vert_id].x_poly))) {
                        num_crossing += 1;
                    }
                }
                vert_id += 1;
                if (polygonVertices[vert_id].x_poly == polygonVertices[start_poly].x_poly &&
                    polygonVertices[vert_id].y_poly == polygonVertices[start_poly].y_poly) {
                    vert_id += 1;
                    start_poly = vert_id;
                }
            }
            
            // if num_crossing is odd = cell is oustside of the polygon
            // if num_crossing is even = cell is inside of the polygon
            if ( (num_crossing%2) != 0 ) {
                int icell_cent_2d = i + j*(WGD->nx-1);
                int icell_canopy_2d = (i+1-i_start) + (j+1-j_start)*nx_canopy;
                
                // Define start index of the canopy in z-direction
                for (auto k=1; k<WGD->z.size(); k++) {
                    if (WGD->terrain[icell_cent_2d]+base_height <= WGD->z[k]) {
                        canopy_bot_index[icell_canopy_2d] = k;
                        canopy_bot[icell_canopy_2d] = WGD->terrain[icell_cent_2d]+base_height;
                        break;
                    }
                }
                
                // Define end index of the canopy in z-direction   
                for (auto k=0; k<WGD->z.size(); k++) {
                    if(WGD->terrain[icell_cent_2d]+base_height+H < WGD->z[k+1]) {
                        canopy_top_index[icell_canopy_2d] = k;
                        canopy_top[icell_canopy_2d] = WGD->terrain[icell_cent_2d]+base_height+H;
                        break;
                    }
                }
                
                // Define hieght of the canopy base in z-direction   
                for (auto k=1; k<WGD->z.size(); k++) {
                    int icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
                    if( WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                        canopy_base[icell_canopy_2d] = WGD->z_face[k-1];
                        break;
                    }
                }
                
                canopy_height[icell_canopy_2d] = canopy_top[icell_canopy_2d]-canopy_bot[icell_canopy_2d];
                
                // define icellflag @ (x,y) for all z(k) in [k_start...k_end]
                for (auto k=canopy_bot_index[icell_canopy_2d]; k<=canopy_top_index[icell_canopy_2d]; k++) {
                    int icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
                    if( WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                        // Canopy cell
                        WGD->icellflag[icell_cent] = cellFlagToUse;             
                    }
                }
            } // end define icellflag!
        }
    }
    
    // find k_start and k_end
    k_start=WGD->nz;
    k_end=0;
    for (auto j=0; j<ny_canopy; j++) {
        for (auto i=0; i<nx_canopy; i++) {
            int icell_canopy_2d = i + j*nx_canopy;
            if( canopy_bot_index[icell_canopy_2d] < k_start )
                k_start = canopy_bot_index[icell_canopy_2d];
            if( canopy_top_index[icell_canopy_2d] > k_end )
                k_end = canopy_top_index[icell_canopy_2d];
        }
    }
    
    // number of point in z -> + 2 (1 ghost below, 1 ghost above)
    // the canopy subarray start at the ground (with bottom ghost cell below the ground) 
    nz_canopy = k_end+2;
    // k_end to match definition used by all building (top face and not top cell)
    k_end++;

    // number of cell-center elements (3D)
    numcell_cent_3d = nx_canopy*ny_canopy*nz_canopy;
    // Resize the canopy-related vectors
    canopy_atten.resize( numcell_cent_3d, 0.0 );
    
    return;
}

void Canopy::canopyRegression(WINDSGeneralData* WGD)
{
  
    int k_top, counter;
    float sum_x, sum_y, sum_xy, sum_x_sq, local_mag;
    float y, xm, ym;
    
    for (auto j=0; j<ny_canopy; j++) {
        for (auto i=0; i<nx_canopy; i++) {
            int id = i+j*nx_canopy;
            if (canopy_top_index[id] > 0) {
                for (auto k=canopy_top_index[id]; k<WGD->nz-2; k++) {
                    k_top = k;
                    if (2*canopy_top[id] < WGD->z[k+1])
                        break;
                }
                if (k_top == canopy_top_index[id]) {
                    k_top = canopy_top_index[id]+1;
                }
                if (k_top > WGD->nz-1) {
                    k_top = WGD->nz-1;
                }
                sum_x = 0;
                sum_y = 0;
                sum_xy = 0;
                sum_x_sq = 0;
                counter = 0;
                for (auto k=canopy_top_index[id]; k<=k_top; k++) {
                    counter +=1;
                    int icell_face = (i-1+i_start) + (j-1+j_start)*WGD->nx + k*WGD->nx*WGD->ny;
                    local_mag = sqrt(pow(WGD->u0[icell_face],2.0)+pow(WGD->v0[icell_face],2.0));
                    y = log(WGD->z[k]);
                    sum_x += local_mag;
                    sum_y += y;
                    sum_xy += local_mag*y;
                    sum_x_sq += pow(local_mag,2.0);
                }
                
                canopy_ustar[id] = WGD->vk*(((counter*sum_x_sq)-pow(sum_x,2.0))/((counter*sum_xy)-(sum_x*sum_y)));
                xm = sum_x/counter;
                ym = sum_y/counter;
                canopy_z0[id] = exp(ym-((WGD->vk/canopy_ustar[id]))*xm);
                
            } // end of if (canopy_top_index[id] > 0)
        }
    } 
    
    return;
}

float Canopy::canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m)
{
    int iter;
    float  uhc, d, d1, d2;
    float tol, fnew, fi;

    tol = z0/100;
    fnew = tol*10;

    d1 = z0;
    d2 = canopy_top;
    d = (d1+d2)/2;

    uhc = (ustar/vk)*(log((canopy_top-d1)/z0)+psi_m);
    fi = ((canopy_atten*uhc*vk)/ustar)-canopy_top/(canopy_top-d1);

    if (canopy_atten > 0) {
        iter = 0;
        while (iter < 200 && abs(fnew) > tol && d < canopy_top && d > z0)
        {
            iter += 1;
            d = (d1+d2)/2;
            uhc = (ustar/vk)*(log((canopy_top-d)/z0)+psi_m);
            fnew = ((canopy_atten*uhc*vk)/ustar) - canopy_top/(canopy_top-d);
            if(fnew*fi>0) {
                d1 = d;
            } else if(fnew*fi<0) {
                d2 = d;
            }
        }
        if (d > canopy_top) {
            d = 10000;
        }
        
    } else {
        d = 0.99*canopy_top;
    }

    return d;
}

float Canopy::canopySlopeMatch(float z0, float canopy_top, float canopy_atten)
{
  
    int iter;
    float tol, d, d1, d2, f;
  
    tol = z0/100;
    // f is the root of the equation (to find d)
    // log[(H-d)/z0] = H/[a(H-d)] 
    f = tol*10;
  
    // initial bound for bisection method (d1,d2)
    // d1 min displacement possible
    // d2 max displacement possible - canopy top
    if (z0 < canopy_top) {
        d1 = z0;
    } else if (z0 > canopy_top) {
        d1 = 0.1;
    }
    d2 = canopy_top;
    d = (d1+d2)/2;
  
    if (canopy_atten > 0) {
        iter = 0;
        // bisection method to find the displacement height
        while (iter < 200 && abs(f) > tol && d < canopy_top && d > z0) {
            iter += 1;
            d = (d1+d2)/2;
            f = log ((canopy_top-d)/z0) - (canopy_top/(canopy_atten*(canopy_top-d)));
            if(f > 0) {
                d1 = d;
            } else if(f<0) {
                d2 = d;
            }
        }
        // if displacement found higher that canopy top => shifted down
        if (d > canopy_top) {
            d = 0.7*canopy_top;
        }
    } else {
        // return this if attenuation coeff is 0.
        d = 10000;
    }
  
    // return displacement height
    return d;
}
