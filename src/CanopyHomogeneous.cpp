#include "CanopyHomogeneous.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

// set et attenuation coefficient 
void CanopyHomogeneous::setCellFlags (const WINDSInputData* WID, WINDSGeneralData* WGD, int building_id)
{
    // When THIS canopy calls this function, we need to do the
    // following:
    //readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
    //canopy_atten, canopy_top);
    
    // this function need to be called to defined the boundary of the canopy and the icellflags
    setCanopyGrid(WGD,building_id);
    
    // Resize the canopy-related vectors
    canopy_atten.resize( numcell_cent_3d, 0.0 );
    
    for (auto j=0; j<ny_canopy; j++) {
        for (auto i=0; i<nx_canopy; i++) {
            int icell_2d = i + j*nx_canopy;
            for (auto k=canopy_bot_index[icell_2d]; k<canopy_top_index[icell_2d]; k++) {
                int icell_3d = i + j*nx_canopy + k*nx_canopy*ny_canopy;
                // initiate all attenuation coefficients to the canopy coefficient
                canopy_atten[icell_3d] = attenuationCoeff;     
            }
        }
    }
    
    return;
}


void CanopyHomogeneous::canopyVegetation(WINDSGeneralData* WGD, int building_id)
{ 
    // Apply canopy parameterization
    canopyCioncoParam(WGD);		
    
    return;
}

void CanopyHomogeneous::canopyWake(WINDSGeneralData* WGD, int building_id)
{
    return;
}

