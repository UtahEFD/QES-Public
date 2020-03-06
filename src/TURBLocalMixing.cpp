#include "TURBLocalMixing.h"
#include "TURBGeneralData.h"

void TURBLocalMixing::defineLength(URBGeneralData *UGD,TURBGeneralData *TGD) {
  
    float vonKar=0.41;

    int nx = TGD->nx;
    int ny = TGD->ny;
    int nz = TGD->nz;

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
                     max_z < TGD->z_cc[k]) {
                    max_z=TGD->z_cc[k];
                }
            }
        }
    }
    // maximum height of local mixing length = 2*max z of objects
    int max_height=nz-2;
    for(int k=0;k<nz;++k){
        if(TGD->z_cc[k]>2.0*max_z){
            max_height=k;
            break;
        }
    }
  
    //seeding Local Mixing Length with the verical distance to the terrain (up to 2*max_z)
    for(size_t id=0;id<TGD->icellfluid.size();id++) {
        int id_cc=TGD->icellfluid[id];
        int k = (id_cc / ((nx-1)*(ny-1)));
        int j = (id_cc - k*(nx-1)*(ny-1))/(nx-1);
        int i = id_cc -  j*(nx-1) - k*(nx-1)*(ny-1);
    
        if(k<max_height)
            TGD->Lm[id_cc] = vonKar*abs(TGD->z_cc[k]-UGD->terrain[i + j*(nx-1)]);
        else
            TGD->Lm[id_cc] = vonKar*(TGD->z_cc[k]);
    }

    getMinDistWall(UGD,TGD,max_height);
  
    //linear interpolation between 2.0*max_z and 2.4*max_z
    std::cout << "[MixLength] \t linear interp of mixing length" << std::endl;
    int k1 = std::min(max_height-1,nz-2);
    int k2 = std::min(k1+k1/5,nz-2);
    for(int i=1;i<nx-2;++i){
        for(int j=1;j<ny-2;++j){
            int id1 = i + j*(nx-1) + k1*(nx-1)*(ny-1);
            int id2 = i + j*(nx-1) + k2*(nx-1)*(ny-1);
            // slope m = (L(z2)-L(z1))/(z2-z1)
            float slope=(TGD->Lm[id2]-TGD->Lm[id1])/(TGD->z_cc[k2]-TGD->z_cc[k1]);
            // linear interp: L(z) = L(z1) + m*(z-z1)
            for(int k=k1;k<k2;++k){
                int id_cc=i + j*(nx-1) + k*(nx-1)*(ny-1);
                TGD->Lm[id_cc]=TGD->Lm[id1]+(TGD->z_cc[k]-TGD->z_cc[k1])*slope;
            }
        }
    }
}

void TURBLocalMixing::getMinDistWall(URBGeneralData *UGD,TURBGeneralData *TGD,int max_height) {

    float vonKar=0.41;

    int nx = TGD->nx;
    int ny = TGD->ny;
    int nz = TGD->nz;

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
                        UGD->icellflag[icell_cent-(nx-1)]==2) {
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

        TGD->Lm[id_cc]=vonKar*(TGD->z_cc.at(k)-TGD->z_fc.at(k));

        float x1 = TGD->x_cc.at(i);
        float y1 = TGD->y_cc.at(j);
        float z1 = TGD->z_fc.at(k);

        if(UGD->icellflag[idxp]==2 && UGD->icellflag[idxm]==2 &&
           UGD->icellflag[idyp]==2 && UGD->icellflag[idym]==2) {
            // terrain on all 4 corner -> nothing to do
        }else if(UGD->icellflag[idxp]==0 && UGD->icellflag[idxm]==0 &&
                 UGD->icellflag[idyp]==0 && UGD->icellflag[idym]==0) {
            // building on all 4 corner -> propagate verically
            for (int kk=0; kk<=maxdist; kk++) {        
                int id=i + j*(nx-1) + (kk+k)*(nx-1)*(ny-1);
                float x2 = TGD->x_cc.at(i);
                float y2 = TGD->y_cc.at(j);
                float z2 = TGD->z_cc.at(kk+k);
                float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                TGD->Lm[id]=std::min(vonKar*dist,TGD->Lm[id]);
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
                        float x2 = TGD->x_cc.at(ii);
                        float y2 = TGD->y_cc.at(jj);
                        float z2 = TGD->z_cc.at(kk+k);
                        float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                        TGD->Lm[id]=std::min(vonKar*dist,TGD->Lm[id]);
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

        TGD->Lm[id_cc]=vonKar*(TGD->x_cc.at(i)-TGD->x_fc.at(i));

        float x1 = TGD->x_fc.at(i);
        float y1 = TGD->y_cc.at(j);
        float z1 = TGD->z_cc.at(k);

        for (int ii=0; ii<=maxdist; ii++) {

            //int k1 = std::max(k,0);
            //int k2 = std::min(k+ii+1,nz-2);
            int j1 = std::max(j-ii,0);
            int j2 = std::min(j+ii+1,ny-2);
            int k1(k),k2(k);


            for (int jj=j1; jj<=j2; jj++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=(i+ii) + jj*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = TGD->x_cc.at(ii+i);
                    float y2 = TGD->y_cc.at(jj);
                    float z2 = TGD->z_cc.at(kk);
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    TGD->Lm[id]=std::min(vonKar*dist,TGD->Lm[id]);
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

        TGD->Lm[id_cc]=vonKar*(TGD->x_fc.at(i+1)-TGD->x_cc.at(i));

        float x1 = TGD->x_fc.at(i+1);
        float y1 = TGD->y_cc.at(j);
        float z1 = TGD->z_cc.at(k);

        for (int ii=0; ii>=-maxdist; ii--) {
      
            //int k1 = std::max(k,0);
            //int k2 = std::min(k-ii+1,nz-2);
            int j1 = std::max(j+ii,0);
            int j2 = std::min(j-ii+1,ny-2);
            int k1(k),k2(k);
      
            for (int jj=j1; jj<=j2; jj++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=(i+1+ii) + jj*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = TGD->x_cc.at(i+ii+1);
                    float y2 = TGD->y_cc.at(jj);
                    float z2 = TGD->z_cc.at(kk);
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    TGD->Lm[id]=std::min(vonKar*dist,TGD->Lm[id]);
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

        TGD->Lm[id_cc]=vonKar*(TGD->y_cc.at(j)-TGD->y_fc.at(j));

        float x1 = TGD->x_cc.at(i);
        float y1 = TGD->y_fc.at(j);
        float z1 = TGD->z_cc.at(k);

        for (int jj=0; jj<=maxdist; jj++) {

            //int k1 = std::max(k,0);
            //int k2 = std::min(k+jj+1,nz-2);
            int i1 = std::max(i-jj-1,0);
            int i2 = std::min(i+jj+1,nx-2);
            int k1(k),k2(k);

            for (int ii=i1; ii<=i2; ii++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=ii + (j+jj)*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = TGD->x_cc.at(ii);
                    float y2 = TGD->y_cc.at(j+jj);
                    float z2 = TGD->z_cc.at(kk);
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    TGD->Lm[id]=std::min(vonKar*dist,TGD->Lm[id]);
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

        TGD->Lm[id_cc]=vonKar*(TGD->y_fc.at(j+1)-TGD->y_cc.at(j));

        float x1 = TGD->x_cc.at(i);
        float y1 = TGD->y_fc.at(j+1);
        float z1 = TGD->z_cc.at(k);

        for (int jj=0; jj>=-maxdist; jj--) {

            //int k1 = std::max(k,0);
            //int k2 = std::min(k-jj+1,nz-2);
            int i1 = std::max(i+jj-1,0);
            int i2 = std::min(i-jj+1,nx-2);

            int k1(k),k2(k);

            for (int ii=i1; ii<=i2; ii++) {
                for (int kk=k1; kk<=k2; kk++) {
                    int id=ii + (j+1+jj)*(nx-1) + (kk)*(nx-1)*(ny-1);
                    float x2 = TGD->x_cc.at(ii);
                    float y2 = TGD->y_cc.at(j+1+jj);
                    float z2 = TGD->z_cc.at(kk);
                    float dist = sqrt(pow((x2-x1),2)+pow((y2-y1),2)+pow((z2-z1),2));
                    TGD->Lm[id]=std::min(vonKar*dist,TGD->Lm[id]);
                }
            }
        }
    }
    std::cout <<"[MixLength] \t cells with wall to the left: DONE " << std::endl;
}

