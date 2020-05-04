#include "Wall.h"

#include "URBGeneralData.h"

void Wall::defineWalls(URBGeneralData *UGD)
{

    float dx = UGD->dx;
    float dy = UGD->dy;
    float dz = UGD->dz;
    int nx = UGD->nx;
    int ny = UGD->ny;
    int nz = UGD->nz;

    for (auto i=0; i<nx-1; i++) {
        for (auto j=0; j<ny-1; j++) {
            for (auto k=1; k<nz-2; k++) {

                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                int icell_face = i + j*nx + k*nx*ny;

                if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                {
                    /// Wall below
                    if (UGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 0 || UGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 2)
                    {
                        UGD->wall_below_indices.push_back(icell_face);
                    }
                    /// Wall above
                    if (UGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 0 || UGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 2)
                    {
                        UGD->wall_above_indices.push_back(icell_face);
                    }
                    /// Wall in back
                    if (UGD->icellflag[icell_cent-1] == 0 || UGD->icellflag[icell_cent-1] == 2)
                    {
                        if (i>0) {
                            UGD->wall_back_indices.push_back(icell_face);
                        }
                    }
                    /// Wall in front
                    if (UGD->icellflag[icell_cent+1] == 0 || UGD->icellflag[icell_cent+1] == 2)
                    {
                        UGD->wall_front_indices.push_back(icell_face);
                    }
                    /// Wall on right
                    if (UGD->icellflag[icell_cent-(nx-1)] == 0 || UGD->icellflag[icell_cent-(nx-1)] == 2)
                    {
                        if (j>0) {
                            UGD->wall_right_indices.push_back(icell_face);
                        }
                    }
                    /// Wall on left
                    if (UGD->icellflag[icell_cent+(nx-1)] == 0 || UGD->icellflag[icell_cent+(nx-1)] == 2)
                    {
                        UGD->wall_left_indices.push_back(icell_face);
                    }
                }
            }
        }
    }

    return; 
}
