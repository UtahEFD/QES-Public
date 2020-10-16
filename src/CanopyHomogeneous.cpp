#include "CanopyHomogeneous.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

// set et attenuation coefficient 
void CanopyHomogeneous::canopyInitial(WINDSGeneralData *WGD)
{
    // When THIS canopy calls this function, we need to do the
    // following:
    //readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
    //canopy_atten, canopy_top);
    
    // this function need to be called to defined the boundary of the canopy and the icellflags
    canopyDefineBoundary(WGD,cellFlagCionco);
    
    for (auto j=0; j<ny_canopy; j++) {
        for (auto i=0; i<nx_canopy; i++) {
            int icell_2d = i + j*nx_canopy;
            for (auto k=canopy_bot_index[icell_2d]; k<=canopy_top_index[icell_2d]; k++) {
                int icell_3d = i + j*nx_canopy + k*nx_canopy*ny_canopy;
                // initiate all attenuation coefficients to the canopy coefficient
                canopy_atten[icell_3d] = attenuationCoeff;     
            }
        }
    }
    
    return;
}


void CanopyHomogeneous::canopyVegetation(WINDSGeneralData* WGD)
{
  
    // Apply canopy parameterization
    canopyParam(WGD);		
    
    return;
}


// Function to apply the urban canopy parameterization
// Based on the version contain Lucas Ulmer's modifications
void CanopyHomogeneous::canopyParam(WINDSGeneralData* WGD)
{
  
    float avg_atten;     /**< average attenuation of the canopy */
    float veg_vel_frac;  /**< vegetation velocity fraction */
    int num_atten;
  
    // Call regression to define ustar and surface roughness of the canopy
    canopyRegression(WGD);

    for (auto j=0; j<ny_canopy; j++) {
        for (auto i=0; i<nx_canopy; i++) {
            int icell_2d = i+j*nx_canopy;

            if (canopy_top[icell_2d] > 0) {
                int icell_3d = i+j*nx_canopy+canopy_top_index[icell_2d]*nx_canopy*ny_canopy;

                // Call the bisection method to find the root
                canopy_d[icell_2d] = canopyBisection(canopy_ustar[icell_2d],canopy_z0[icell_2d],
                                                     canopy_height[icell_2d],canopy_atten[icell_3d],WGD->vk,0.0);
                //std::cout << "WGD->vk:" << WGD->vk << "\n";
                //std::cout << "WGD->canopy_atten[icell_cent]:" << WGD->canopy_atten[icell_cent] << "\n";
                if (canopy_d[icell_2d] == 10000) {
                    std::cout << "bisection failed to converge" << "\n";
                    canopy_d[icell_2d] = canopySlopeMatch(canopy_z0[icell_2d],canopy_height[icell_2d],
                                                          canopy_atten[icell_3d]);
                }
                
                /**< velocity at the height of the canopy */
                // Local variable - not being used by anything... so
                // commented out for now.
                //
                //float u_H = (WGD->canopy_ustar[id]/WGD->vk)*
                //  log((WGD->canopy_top[id]-WGD->canopy_d[id])/WGD->canopy_z0[id]);
                
                for (auto k=1; k < WGD->nz-1; k++) {
                    int icell_face = (i-1+i_start) + (j-1+j_start)*WGD->nx + k*WGD->nx*WGD->ny;
                    float z_rel = WGD->z[k] - canopy_base[icell_2d];
                    
                    if(WGD->z[k] < canopy_base[icell_2d]) {
                        // below the terrain or building
                    } else if (WGD->z[k] < canopy_top[icell_2d]) {
                        if (canopy_atten[icell_3d] > 0) {
                            icell_3d = i+j*nx_canopy+k*nx_canopy*ny_canopy;
                            avg_atten = canopy_atten[icell_3d];
                            
                            
                            if( canopy_atten[icell_3d+nx_canopy*ny_canopy]!=canopy_atten[icell_3d] ||
                                canopy_atten[icell_3d-nx_canopy*ny_canopy]!=canopy_atten[icell_3d] ) {
                                num_atten = 1;
                                if( canopy_atten[icell_3d+nx_canopy*ny_canopy] > 0 ) {
                                    avg_atten += canopy_atten[icell_3d+nx_canopy*ny_canopy];
                                    num_atten += 1;
                                }
                                if( canopy_atten[icell_3d-nx_canopy*ny_canopy] > 0 ) {
                                    avg_atten += canopy_atten[icell_3d-nx_canopy*ny_canopy];
                                    num_atten += 1;
                                }
                                avg_atten /= num_atten;
                            }
                            
                            /*
                            veg_vel_frac = log((canopy_top[icell_2d] - canopy_d[icell_2d])/
                                               canopy_z0[icell_2d])*exp(avg_atten*((WGD->z[k]/canopy_top[icell_2d])-1))/
                                log(WGD->z[k]/canopy_z0[icell_2d]);
                            */
                            
                            // correction on the velocity within the canopy
                            veg_vel_frac = log((canopy_height[icell_2d] - canopy_d[icell_2d])/
                                               canopy_z0[icell_2d])*exp(avg_atten*((z_rel/canopy_height[icell_2d])-1))/
                                log(z_rel/canopy_z0[icell_2d]);
                            // check if correction is bound and well defined
                            if (veg_vel_frac > 1 || veg_vel_frac < 0) {
                                veg_vel_frac = 1; 
                            }
                            
                            WGD->u0[icell_face] *= veg_vel_frac;
                            WGD->v0[icell_face] *= veg_vel_frac;
                                     
                            // at the edge of the canopy need to adjust velocity at the next face 
                            // use canopy_top to detect the edge (worke with level changes)
                            if (j-1+j_start < WGD->ny-2) {
                                if (canopy_top[icell_2d+nx_canopy] == 0.0) {
                                    WGD->v0[icell_face+WGD->nx] *= veg_vel_frac;
                                }
                            }
                            if (i-1+i_start < WGD->nx-2) {
                                if(canopy_top[icell_2d+1] == 0.0) {
                                    WGD->u0[icell_face+1] *= veg_vel_frac;
                                }
                            }
                        }
                    } else {
                        // correction on the velocity above the canopy
                        veg_vel_frac = log((z_rel-canopy_d[icell_2d])/canopy_z0[icell_2d])/
                            log(z_rel/canopy_z0[icell_2d]);
                        // check if correction is bound and well defined
                        if (veg_vel_frac > 1 || veg_vel_frac < 0)
                        {
                            veg_vel_frac = 1;
                        }
                        
                        WGD->u0[icell_face] *= veg_vel_frac;
                        WGD->v0[icell_face] *= veg_vel_frac;

                        // at the edge of the canopy need to adjust velocity at the next face 
                        // use canopy_top to detect the edge (worke with level changes)
                        if (j-1+j_start < WGD->ny-2) {
                            icell_3d = i+j*nx_canopy+canopy_bot_index[icell_2d]*nx_canopy*ny_canopy;
                            if(canopy_top[icell_2d+nx_canopy] == 0.0) {
                                WGD->v0[icell_face+WGD->nx] *= veg_vel_frac;
                            }
                        }
                        if (i-1+i_start < WGD->nx-2) {
                            icell_3d = i+j*nx_canopy+canopy_bot_index[icell_2d]*nx_canopy*ny_canopy;
                            if (canopy_top[icell_2d+1] == 0.0) {
                                WGD->u0[icell_face+1] *= veg_vel_frac;
                            }
                        }
                    }
                } // end of for(auto k=1; k < WGD->nz-1; k++)
            }
        }
    }
    
    return;
}


