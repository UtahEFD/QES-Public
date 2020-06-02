#include "LocalMixingDefault.h"

// These take care of the circular reference
#include "URBInputData.h"
#include "URBGeneralData.h"

void LocalMixingDefault::defineMixingLength(const URBInputData* UID,URBGeneralData* UGD) 
{
    int nx = UGD->nx;
    int ny = UGD->ny;
    int nz = UGD->nz;
    
    // z cell-center
    std::vector<float> z_cc;
    z_cc.resize(nz-1, 0);
    z_cc = UGD->z;
    
    //seeding Local Mixing Length with the verical distance to the terrain (up to 2*max_z)
    for (int i=0; i<nx-1; i++) {
        for (int j=0; j<ny-1; j++) {
            for (int k=1; k<nz-2; k++) {
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2) {
                    UGD->mixingLengths[icell_cent] = z_cc[k]-UGD->terrain[i + j*(nx-1)];
                } 
                if(UGD->mixingLengths[icell_cent] < 0.0) {
                    UGD->mixingLengths[icell_cent] = 0.0;
                }
            }
        }
    }
    
    return;
}

