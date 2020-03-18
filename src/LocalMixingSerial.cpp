#include "LocalMixingSerial.h"

void LocalMixingSerial::defineMixingLength(URBGeneralData *UGD) {
  
    //float vonKar=0.41;

    int nx = UGD->nx;
    int ny = UGD->ny;
    int nz = UGD->nz;
    
    float dz = UGD->dz;
    float dy = UGD->dy;
    float dx = UGD->dx;

    // x-grid (face-center & cell-center)
    x_fc.resize(nx, 0);
    x_cc.resize(nx-1, 0);

    // y-grid (face-center & cell-center)
    y_fc.resize(ny, 0);
    y_cc.resize(ny-1, 0);

    // z-grid (face-center & cell-center)
    z_fc.resize(nz, 0);
    z_cc.resize(nz-1, 0);

    /*
      The face-center x,y,z are defined here as a place holder, will have to be imported
      form URB when non-uniform grid is used
    */

    // x face-center
    for(int i=1;i<nx-1;i++) {
        x_fc[i]= 0.5*(UGD->x[i-1]+UGD->x[i]);
    }
    x_fc[0] = x_fc[1]-dx;
    x_fc[nx-1] = x_fc[nx-2]+dx;
    // x cell-center
    x_cc = UGD->x;

    // y face-center
    for(int i=1;i<ny-1;i++) {
        y_fc[i] = 0.5*(UGD->y[i-1]+UGD->y[i]);
    }
    y_fc[0] = y_fc[1]-dy;
    y_fc[ny-1] = y_fc[ny-2]+dy;
    // y cell-center
    y_cc = UGD->y;

    // z face-center (with ghost cell under the ground)
    for(int i=1;i<nz-1;i++) {
        z_fc[i] = 0.5*(UGD->z[i-1]+UGD->z[i]);
    }
    z_fc[0] = z_fc[1]-dz;
    z_fc[nz-1] = z_fc[nz-2]+dz;
    // z cell-center
    z_cc = UGD->z;

    // find max height of solid objects in the domaine (max_z)
    /*
      [FM] this works only with the terrain
      //float max_z=*std::max_element(UGD->terrain.begin(),UGD->terrain.end());  
      the following code works based on the icellflag -> works for both the 
      terrain and the buildings
    */
    float max_z=0;
    for (int i=1; i<nx-2; i++) {
        for (int j=1; j<ny-2; j++) {
            for (int k=0; k<nz-2; k++) {
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                if ( (UGD->icellflag[icell_cent] ==0 || UGD->icellflag[icell_cent] ==2) &&
                     max_z < z_cc[k]) {
                    max_z=z_cc[k];
                }
            }
        }
    }
    
    // maximum height of local mixing length = 2*max z of objects
    int max_height=nz-2;
    for(int k=0;k<nz;++k) {
        if(z_cc[k]>2.0*max_z) {
            max_height=k;
            break;
        }
    }
  
    //seeding Local Mixing Length with the verical distance to the terrain (up to 2*max_z)
    for (int i=1; i<nx-2; i++) {
        for (int j=1; j<ny-2; j++) {
            for (int k=0; k<nz-2; k++) {
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                if ( (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2) ) {
                    if(k<max_height) {
                        UGD->mixingLengths[icell_cent] = abs(z_cc[k]-UGD->terrain[i + j*(nx-1)]);
                    } else {
                        UGD->mixingLengths[icell_cent] = z_cc[k];
                    }
                    if(UGD->mixingLengths[icell_cent] < 0) {
                        UGD->mixingLengths[icell_cent] = 0;
                    }
                }
            }
        }
    }
    
    getMinDistWall(UGD,max_height);
  
    //linear interpolation between 2.0*max_z and 2.4*max_z
    std::cout << "[MixLength] \t linear interp of mixing length" << std::endl;
    int k1 = std::min(max_height-1,nz-2);
    int k2 = std::min(k1+k1/5,nz-2);
    for(int i=1;i<nx-2;++i){
        for(int j=1;j<ny-2;++j){
            int id1 = i + j*(nx-1) + k1*(nx-1)*(ny-1);
            int id2 = i + j*(nx-1) + k2*(nx-1)*(ny-1);
            // slope m = (L(z2)-L(z1))/(z2-z1)
            float slope=(UGD->mixingLengths[id2]-UGD->mixingLengths[id1])/(z_cc[k2]-z_cc[k1]);
            // linear interp: L(z) = L(z1) + m*(z-z1)
            for(int k=k1;k<k2;++k){
                int id_cc=i + j*(nx-1) + k*(nx-1)*(ny-1);
                UGD->mixingLengths[id_cc]=UGD->mixingLengths[id1]+(z_cc[k]-z_cc[k1])*slope;
            }
        }
    }
}

void LocalMixingSerial::getMinDistWall(URBGeneralData *UGD,int max_height) {

    float vonKar=0.41;

    int nx = UGD->nx;
    int ny = UGD->ny;
    int nz = UGD->nz;

    // defining the walls
    for (int i=1; i<nx-2; i++) {
        for (int j=1; j<ny-2; j++) {
            for (int k=1; k<nz-2; k++) {
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
        
                if (UGD->icellflag[icell_cent] !=0 && UGD->icellflag[icell_cent] !=2) {
                    /// Wall below
                    if (UGD->icellflag[icell_cent-(nx-1)*(ny-1)]==0 || 
                        UGD->icellflag[icell_cent-(nx-1)*(ny-1)]==2) 
                    {
                        wall_below_indices.push_back(icell_cent);
                    }
                    /// Wall above
                    if (UGD->icellflag[icell_cent+(nx-1)*(ny-1)]==0 || 
                        UGD->icellflag[icell_cent+(nx-1)*(ny-1)]==2) 
                    {
                        wall_above_indices.push_back(icell_cent);
                    }
                    /// Wall in back
                    if (UGD->icellflag[icell_cent-1]==0 || 
                        UGD->icellflag[icell_cent-1]==2) 
                    {
                        wall_back_indices.push_back(icell_cent);
                    }
                    /// Wall in front
                    if (UGD->icellflag[icell_cent+1]==0 || 
                        UGD->icellflag[icell_cent+1]==2) 
                    {
                        wall_front_indices.push_back(icell_cent);
                    }
                    /// Wall on right
                    if (UGD->icellflag[icell_cent-(nx-1)]==0 || 
                        UGD->icellflag[icell_cent-(nx-1)]==2) 
                    {
                        wall_right_indices.push_back(icell_cent);
                    }
                    /// Wall on left
                    if (UGD->icellflag[icell_cent+(nx-1)]==0 || 
                        UGD->icellflag[icell_cent+(nx-1)]==2) 
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

        UGD->mixingLengths[id_cc]=z_cc.at(k)-z_fc.at(k);

        float x1 = x_cc.at(i);
        float y1 = y_cc.at(j);
        float z1 = z_fc.at(k);

        if(UGD->icellflag[idxp]==2 && UGD->icellflag[idxm]==2 &&
           UGD->icellflag[idyp]==2 && UGD->icellflag[idym]==2) {
            // terrain on all 4 corner -> nothing to do
        }else if(UGD->icellflag[idxp]==0 && UGD->icellflag[idxm]==0 &&
                 UGD->icellflag[idyp]==0 && UGD->icellflag[idym]==0) {
            // building on all 4 corner -> propagate verically
            for (int kk=0; kk<=maxdist; kk++) {        
                int id=i + j*(nx-1) + (kk+k)*(nx-1)*(ny-1);
                float x2 = x_cc.at(i);
                float y2 = y_cc.at(j);
                float z2 = z_cc.at(kk+k);
                float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                UGD->mixingLengths[id]=std::min(dist,static_cast<float>(UGD->mixingLengths[id]));
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
                        float x2 = x_cc.at(ii);
                        float y2 = y_cc.at(jj);
                        float z2 = z_cc.at(kk+k);
                        float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                        UGD->mixingLengths[id]=std::min(dist,static_cast<float>(UGD->mixingLengths[id]));
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

        UGD->mixingLengths[id_cc]=x_cc.at(i)-x_fc.at(i);

        float x1 = x_fc.at(i);
        float y1 = y_cc.at(j);
        float z1 = z_cc.at(k);

        for (int ii=0; ii<=maxdist; ii++) {

            //int k1 = std::max(k,0);
            //int k2 = std::min(k+ii+1,nz-2);
            int j1 = std::max(j-ii,0);
            int j2 = std::min(j+ii+1,ny-2);
            int k1(k),k2(k);


            for (int jj=j1; jj<=j2; jj++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=(i+ii) + jj*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = x_cc.at(ii+i);
                    float y2 = y_cc.at(jj);
                    float z2 = z_cc.at(kk);
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    UGD->mixingLengths[id]=std::min(dist,static_cast<float>(UGD->mixingLengths[id]));
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

        UGD->mixingLengths[id_cc]=x_fc.at(i+1)-x_cc.at(i);

        float x1 = x_fc.at(i+1);
        float y1 = y_cc.at(j);
        float z1 = z_cc.at(k);

        for (int ii=0; ii>=-maxdist; ii--) {
      
            //int k1 = std::max(k,0);
            //int k2 = std::min(k-ii+1,nz-2);
            int j1 = std::max(j+ii,0);
            int j2 = std::min(j-ii+1,ny-2);
            int k1(k),k2(k);
      
            for (int jj=j1; jj<=j2; jj++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=(i+1+ii) + jj*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = x_cc.at(i+ii+1);
                    float y2 = y_cc.at(jj);
                    float z2 = z_cc.at(kk);
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    UGD->mixingLengths[id]=std::min(dist,static_cast<float>(UGD->mixingLengths[id]));
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

        UGD->mixingLengths[id_cc]=y_cc.at(j)-y_fc.at(j);

        float x1 = x_cc.at(i);
        float y1 = y_fc.at(j);
        float z1 = z_cc.at(k);

        for (int jj=0; jj<=maxdist; jj++) {

            //int k1 = std::max(k,0);
            //int k2 = std::min(k+jj+1,nz-2);
            int i1 = std::max(i-jj-1,0);
            int i2 = std::min(i+jj+1,nx-2);
            int k1(k),k2(k);

            for (int ii=i1; ii<=i2; ii++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=ii + (j+jj)*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = x_cc.at(ii);
                    float y2 = y_cc.at(j+jj);
                    float z2 = z_cc.at(kk);
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    UGD->mixingLengths[id]=std::min(dist,static_cast<float>(UGD->mixingLengths[id]));
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

        UGD->mixingLengths[id_cc]=y_fc.at(j+1)-y_cc.at(j);

        float x1 = x_cc.at(i);
        float y1 = y_fc.at(j+1);
        float z1 = z_cc.at(k);

        for (int jj=0; jj>=-maxdist; jj--) {

            //int k1 = std::max(k,0);
            //int k2 = std::min(k-jj+1,nz-2);
            int i1 = std::max(i+jj-1,0);
            int i2 = std::min(i-jj+1,nx-2);

            int k1(k),k2(k);

            for (int ii=i1; ii<=i2; ii++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=ii + (j+1+jj)*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = x_cc.at(ii);
                    float y2 = y_cc.at(j+1+jj);
                    float z2 = z_cc.at(kk);
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    UGD->mixingLengths[id]=std::min(dist,static_cast<float>(UGD->mixingLengths[id]));
                }
            }
        }
    }
    std::cout <<"[MixLength] \t cells with wall to the left: DONE " << std::endl;
}

