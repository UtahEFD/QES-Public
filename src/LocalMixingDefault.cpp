#include "LocalMixingDefault.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

void LocalMixingDefault::defineMixingLength(const WINDSInputData* WID,WINDSGeneralData* WGD) 
{
    int nx = WGD->nx;
    int ny = WGD->ny;
    int nz = WGD->nz;

    // z cell-center
    std::vector<float> z_cc;
    z_cc.resize(nz-1, 0);
    z_cc = WGD->z;

    //seeding Local Mixing Length with the verical distance to the terrain (up to 2*max_z)
    for (int i=0; i<nx-1; i++) {
        for (int j=0; j<ny-1; j++) {
            for (int k=1; k<nz-2; k++) {
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                    WGD->mixingLengths[icell_cent] = z_cc[k]-WGD->terrain[i + j*(nx-1)];
                }
                if(WGD->mixingLengths[icell_cent] < 0.0) {
                    WGD->mixingLengths[icell_cent] = 0.0;
                }
            }
        }
    }

    return;
}
