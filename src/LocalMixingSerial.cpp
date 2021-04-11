/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file LocalMixingSerial.cpp
 * @brief :document this:
 * @sa LocalMixing
 */

#include "LocalMixingSerial.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

void LocalMixingSerial::defineMixingLength(const WINDSInputData* WID,WINDSGeneralData* WGD)
{
    int nx = WGD->nx;
    int ny = WGD->ny;
    int nz = WGD->nz;

    float dz = WGD->dz;
    float dy = WGD->dy;
    float dx = WGD->dx;

    // x-grid (face-center & cell-center)
    x_fc.resize(nx, 0);
    x_cc.resize(nx-1, 0);

    // y-grid (face-center & cell-center)
    y_fc.resize(ny, 0);
    y_cc.resize(ny-1, 0);

    // z-grid (face-center & cell-center)
    z_fc.resize(nz, 0);
    z_cc.resize(nz-1, 0);

    // x cell-center
    x_cc = WGD->x;
    // x face-center (this assume constant dx for the moment, same as QES-winds)
    for(int i=1;i<nx-1;i++) {
        x_fc[i]= 0.5*(WGD->x[i-1]+WGD->x[i]);
    }
    x_fc[0] = x_fc[1]-dx;
    x_fc[nx-1] = x_fc[nx-2]+dx;

    // y cell-center
    y_cc = WGD->y;
    // y face-center (this assume constant dy for the moment, same as QES-winds)
    for(int i=1;i<ny-1;i++) {
        y_fc[i] = 0.5*(WGD->y[i-1]+WGD->y[i]);
    }
    y_fc[0] = y_fc[1]-dy;
    y_fc[ny-1] = y_fc[ny-2]+dy;

    // z cell-center
    z_cc = WGD->z;
    // z face-center (with ghost cell under the ground)
    for(int i=1;i<nz;i++) {
        z_fc[i] = WGD->z_face[i-1];
    }
    z_fc[0] = z_fc[1]-dz;

    // find max height of solid objects in the domaine (max_z)
    /*
      [FM] this works only with the terrain
      //float max_z=*std::max_element(WGD->terrain.begin(),WGD->terrain.end());
      the following code works based on the icellflag -> works for both the
      terrain and the buildings
    */
    float max_z=0;
    for (int i=1; i<nx-2; i++) {
        for (int j=1; j<ny-2; j++) {
            for (int k=0; k<nz-2; k++) {
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                if ( (WGD->icellflag[icell_cent] ==0 || WGD->icellflag[icell_cent] ==2) &&
                     max_z < z_cc[k]) {
                    max_z=z_cc[k];
                }
            }
        }
    }

    // maximum height of local mixing length = 2*max z of objects
    int max_height=nz-2;

    for(int k=0;k<nz-1;++k) {
        if(z_cc[k]>3.0*max_z) {
            max_height=k;
            break;
        }
    }

    //seeding Local Mixing Length with the verical distance to the terrain (up to 2*max_z)
    for (int i=1; i<nx-2; i++) {
        for (int j=1; j<ny-2; j++) {
            for (int k=0; k<nz-2; k++) {
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                if ( (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) ) {
                    if(k<max_height) {
                        WGD->mixingLengths[icell_cent] = z_cc[k]-WGD->terrain[i + j*(nx-1)];
                    } else {
                        WGD->mixingLengths[icell_cent] = z_cc[k];
                    }
                    if(WGD->mixingLengths[icell_cent] < 0.0) {
                        WGD->mixingLengths[icell_cent] = 0.0;
                    }
                }
            }
        }
    }

    getMinDistWall(WGD,max_height);

    //linear interpolation between 2.0*max_z and 2.4*max_z
    std::cout << "[MixLength] \t linear interp of mixing length" << std::endl;
    int k1 = std::min(max_height-1,nz-2);
    int k2 = std::min(k1+k1/5,nz-2);
    for(int i=1;i<nx-2;++i){
        for(int j=1;j<ny-2;++j){
            int id1 = i + j*(nx-1) + k1*(nx-1)*(ny-1);
            int id2 = i + j*(nx-1) + k2*(nx-1)*(ny-1);
            // slope m = (L(z2)-L(z1))/(z2-z1)
            float slope=(WGD->mixingLengths[id2]-WGD->mixingLengths[id1])/(z_cc[k2]-z_cc[k1]);
            // linear interp: L(z) = L(z1) + m*(z-z1)
            for(int k=k1;k<k2;++k){
                int id_cc=i + j*(nx-1) + k*(nx-1)*(ny-1);
                WGD->mixingLengths[id_cc]=WGD->mixingLengths[id1]+(z_cc[k]-z_cc[k1])*slope;
            }
        }
    }

    if(WID->turbParams->save2file){
        saveMixingLength(WID,WGD);
    }

    return;
}

void LocalMixingSerial::getMinDistWall(WINDSGeneralData *WGD,int max_height) {

    int nx = WGD->nx;
    int ny = WGD->ny;
    int nz = WGD->nz;

    // defining the walls
    for (int i=1; i<nx-2; i++) {
        for (int j=1; j<ny-2; j++) {
            for (int k=1; k<nz-2; k++) {
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);

                if (WGD->icellflag[icell_cent] !=0 && WGD->icellflag[icell_cent] !=2) {
                    /// Wall below
                    if (WGD->icellflag[icell_cent-(nx-1)*(ny-1)]==0 ||
                        WGD->icellflag[icell_cent-(nx-1)*(ny-1)]==2)
                    {
                        wall_below_indices.push_back(icell_cent);
                    }
                    /// Wall above
                    if (WGD->icellflag[icell_cent+(nx-1)*(ny-1)]==0 ||
                        WGD->icellflag[icell_cent+(nx-1)*(ny-1)]==2)
                    {
                        wall_above_indices.push_back(icell_cent);
                    }
                    /// Wall in back
                    if (WGD->icellflag[icell_cent-1]==0 ||
                        WGD->icellflag[icell_cent-1]==2)
                    {
                        wall_back_indices.push_back(icell_cent);
                    }
                    /// Wall in front
                    if (WGD->icellflag[icell_cent+1]==0 ||
                        WGD->icellflag[icell_cent+1]==2)
                    {
                        wall_front_indices.push_back(icell_cent);
                    }
                    /// Wall on right
                    if (WGD->icellflag[icell_cent-(nx-1)]==0 ||
                        WGD->icellflag[icell_cent-(nx-1)]==2)
                    {
                        wall_right_indices.push_back(icell_cent);
                    }
                    /// Wall on left
                    if (WGD->icellflag[icell_cent+(nx-1)]==0 ||
                        WGD->icellflag[icell_cent+(nx-1)]==2)
                    {
                        wall_left_indices.push_back(icell_cent);
                    }
                }
            }
        }
    }

    std::cout <<"[MixLength] \t cells with wall below: "<< wall_below_indices.size() << std::endl;
    std::cout <<"[MixLength] \t cells with wall above: "<< wall_above_indices.size() << std::endl;
    std::cout <<"[MixLength] \t cells with wall in the front: "<< wall_front_indices.size() << std::endl;
    std::cout <<"[MixLength] \t cells with wall in the back: "<< wall_back_indices.size() << std::endl;
    std::cout <<"[MixLength] \t cells with wall to the right: "<< wall_right_indices.size() << std::endl;
    std::cout <<"[MixLength] \t cells with wall to the left: "<< wall_left_indices.size() << std::endl;

    // apply mixing length to the cells with wall below
    for (size_t id=0; id < wall_below_indices.size(); id++){

        int id_cc=wall_below_indices[id];
        int idxp=id_cc-(nx-1)*(ny-1)+1;
        int idxm=id_cc-(nx-1)*(ny-1)-1;
        int idyp=id_cc-(nx-1)*(ny-1)+(nx-1);
        int idym=id_cc-(nx-1)*(ny-1)-(nx-1);

        int k = (int)(id_cc / ((nx-1)*(ny-1)));
        int j = (int)((id_cc - k*(nx-1)*(ny-1))/(nx-1));
        int i = id_cc -  j*(nx-1) - k*(nx-1)*(ny-1);
        int maxdist=max_height-k;

        WGD->mixingLengths[id_cc]=z_cc[k]-z_fc[k];

        float x1 = x_cc[i];
        float y1 = y_cc[j];
        float z1 = z_fc[k];

        if(WGD->icellflag[idxp]==2 && WGD->icellflag[idxm]==2 &&
           WGD->icellflag[idyp]==2 && WGD->icellflag[idym]==2) {
            // terrain on all 4 corner -> nothing to do
        }else if(WGD->icellflag[idxp]==0 && WGD->icellflag[idxm]==0 &&
                 WGD->icellflag[idyp]==0 && WGD->icellflag[idym]==0) {
            // building on all 4 corner -> propagate verically
            for (int kk=0; kk<=maxdist; kk++) {
                int id=i + j*(nx-1) + (kk+k)*(nx-1)*(ny-1);
                float x2 = x_cc[i];
                float y2 = y_cc[j];
                float z2 = z_cc[kk+k];
                float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                WGD->mixingLengths[id]=std::min(dist,static_cast<float>(WGD->mixingLengths[id]));
            }
        }else{
            // propagate in all direction
            for (int kk=0; kk<=maxdist; kk++) {

                //propagate to the whole domaine
                int i1 = std::max(i-maxdist-kk,1);
                int i2 = std::min(i+maxdist+kk,nx-2);
                int j1 = std::max(j-maxdist-kk,1);
                int j2 = std::min(j+maxdist+kk,ny-2);

                for (int jj=j1; jj<=j2; jj++) {
                    for (int ii=i1; ii<=i2; ii++) {
                        int id=ii + jj*(nx-1) + (kk+k)*(nx-1)*(ny-1);
                        float x2 = x_cc[ii];
                        float y2 = y_cc[jj];
                        float z2 = z_cc[kk+k];
                        float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                        WGD->mixingLengths[id]=std::min(dist,static_cast<float>(WGD->mixingLengths[id]));
                    }
                }
            }
        }
    }
    std::cout <<"[MixLength] \t cells with wall below: DONE " << std::endl;

    /// apply mixing length to the cells with wall in back
    for (size_t id=0; id < wall_back_indices.size(); id++){

        int id_cc=wall_back_indices[id];
        int k = (int)(id_cc / ((nx-1)*(ny-1)));
        int j = (int)((id_cc - k*(nx-1)*(ny-1))/(nx-1));
        int i = id_cc -  j*(nx-1) - k*(nx-1)*(ny-1);
        int maxdist=0;
        if(i+k<nx-1){
            maxdist=k;
        } else {
            maxdist=nx-1-i;
        }

        WGD->mixingLengths[id_cc]=x_cc.at(i)-x_fc.at(i);

        float x1 = x_fc[i];
        float y1 = y_cc[j];
        float z1 = z_cc[k];

        for (int ii=0; ii<=maxdist; ii++) {

            //int k1 = std::max(k,0);
            //int k2 = std::min(k+ii+1,nz-2);
            int j1 = std::max(j-ii,0);
            int j2 = std::min(j+ii+1,ny-2);
            int k1(k),k2(k);


            for (int jj=j1; jj<=j2; jj++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=(i+ii) + jj*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = x_cc[ii+i];
                    float y2 = y_cc[jj];
                    float z2 = z_cc[kk];
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    WGD->mixingLengths[id]=std::min(dist,static_cast<float>(WGD->mixingLengths[id]));
                }
            }
        }
    }
    std::cout <<"[MixLength] \t cells with wall in the  back: DONE " << std::endl;

    /// apply mixing length to the cells with wall in front
    for (size_t id=0; id < wall_front_indices.size(); id++){

        int id_cc=wall_front_indices[id];
        int k = (int)(id_cc / ((nx-1)*(ny-1)));
        int j = (int)((id_cc - k*(nx-1)*(ny-1))/(nx-1));
        int i = id_cc -  j*(nx-1) - k*(nx-1)*(ny-1);
        int maxdist=0;
        if(i-k>0){
            maxdist=k;
        } else {
            maxdist=i;
        }

        WGD->mixingLengths[id_cc]=x_fc.at(i+1)-x_cc.at(i);

        float x1 = x_fc[i+1];
        float y1 = y_cc[j];
        float z1 = z_cc[k];

        for (int ii=0; ii>=-maxdist; ii--) {

            //int k1 = std::max(k,0);
            //int k2 = std::min(k-ii+1,nz-2);
            int j1 = std::max(j+ii,0);
            int j2 = std::min(j-ii+1,ny-2);
            int k1(k),k2(k);

            for (int jj=j1; jj<=j2; jj++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=(i+1+ii) + jj*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = x_cc[i+ii+1];
                    float y2 = y_cc[jj];
                    float z2 = z_cc[kk];
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    WGD->mixingLengths[id]=std::min(dist,static_cast<float>(WGD->mixingLengths[id]));
                }
            }
        }
    }
    std::cout <<"[MixLength] \t cells with wall in the front: DONE " << std::endl;

    /// apply mixing length to the cells with wall to right
    for (size_t id=0; id < wall_right_indices.size(); id++){

        int id_cc=wall_right_indices[id];
        int k = (int)(id_cc / ((nx-1)*(ny-1)));
        int j = (int)((id_cc - k*(nx-1)*(ny-1))/(nx-1));
        int i = id_cc -  j*(nx-1) - k*(nx-1)*(ny-1);
        int maxdist=0;
        if(j+k<ny-1){
            maxdist=k;
        } else {
            maxdist=ny-1-j;
        }

        WGD->mixingLengths[id_cc]=y_cc[j]-y_fc[j];

        float x1 = x_cc[i];
        float y1 = y_fc[j];
        float z1 = z_cc[k];

        for (int jj=0; jj<=maxdist; jj++) {

            //int k1 = std::max(k,0);
            //int k2 = std::min(k+jj+1,nz-2);
            int i1 = std::max(i-jj,0);
            int i2 = std::min(i+jj+1,nx-2);
            int k1(k),k2(k);

            for (int ii=i1; ii<=i2; ii++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=ii + (j+jj)*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = x_cc[ii];
                    float y2 = y_cc[j+jj];
                    float z2 = z_cc[kk];
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    WGD->mixingLengths[id]=std::min(dist,static_cast<float>(WGD->mixingLengths[id]));
                }
            }
        }
    }
    std::cout <<"[MixLength] \t cells with wall to the right: DONE " << std::endl;

    /// apply mixing length to the cells with wall to left
    for (size_t id=0; id < wall_left_indices.size(); id++){

        int id_cc=wall_left_indices[id];
        int k = (int)(id_cc / ((nx-1)*(ny-1)));
        int j = (int)((id_cc - k*(nx-1)*(ny-1))/(nx-1));
        int i = id_cc -  j*(nx-1) - k*(nx-1)*(ny-1);
        int maxdist=0;
        if(j-k>0){
            maxdist=k;
        } else {
            maxdist=j;
        }

        WGD->mixingLengths[id_cc]=y_fc[j+1]-y_cc[j];

        float x1 = x_cc[i];
        float y1 = y_fc[j+1];
        float z1 = z_cc[k];

        for (int jj=0; jj>=-maxdist; jj--) {

            //int k1 = std::max(k,0);
            //int k2 = std::min(k-jj+1,nz-2);
            int i1 = std::max(i+jj,0);
            int i2 = std::min(i-jj+1,nx-2);

            int k1(k),k2(k);

            for (int ii=i1; ii<=i2; ii++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=ii + (j+1+jj)*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = x_cc[ii];
                    float y2 = y_cc[j+1+jj];
                    float z2 = z_cc[kk];
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    WGD->mixingLengths[id]=std::min(dist,static_cast<float>(WGD->mixingLengths[id]));
                }
            }
        }
    }
    std::cout <<"[MixLength] \t cells with wall to the left: DONE " << std::endl;
}
