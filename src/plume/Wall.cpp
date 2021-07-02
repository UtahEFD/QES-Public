#include "Wall.h"

#include "WINDSGeneralData.h"

void Wall::defineWalls(WINDSGeneralData *WGD)
{

    float dx = WGD->dx;
    float dy = WGD->dy;
    float dz = WGD->dz;
    int nx = WGD->nx;
    int ny = WGD->ny;
    int nz = WGD->nz;

    for (auto i=0; i<nx-1; i++) {
        for (auto j=0; j<ny-1; j++) {
            for (auto k=1; k<nz-2; k++) {

                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                int icell_face = i + j*nx + k*nx*ny;

                if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)
                {
                    /// Wall below
                    if (WGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 0 || WGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 2)
                    {
                        WGD->wall_below_indices.push_back(icell_face);
                    }
                    /// Wall above
                    if (WGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 0 || WGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 2)
                    {
                        WGD->wall_above_indices.push_back(icell_face);
                    }
                    /// Wall in back
                    if (WGD->icellflag[icell_cent-1] == 0 || WGD->icellflag[icell_cent-1] == 2)
                    {
                        if (i>0) {
                            WGD->wall_back_indices.push_back(icell_face);
                        }
                    }
                    /// Wall in front
                    if (WGD->icellflag[icell_cent+1] == 0 || WGD->icellflag[icell_cent+1] == 2)
                    {
                        WGD->wall_front_indices.push_back(icell_face);
                    }
                    /// Wall on right
                    if (WGD->icellflag[icell_cent-(nx-1)] == 0 || WGD->icellflag[icell_cent-(nx-1)] == 2)
                    {
                        if (j>0) {
                            WGD->wall_right_indices.push_back(icell_face);
                        }
                    }
                    /// Wall on left
                    if (WGD->icellflag[icell_cent+(nx-1)] == 0 || WGD->icellflag[icell_cent+(nx-1)] == 2)
                    {
                        WGD->wall_left_indices.push_back(icell_face);
                    }
                }
            }
        }
    }

    return; 
}
