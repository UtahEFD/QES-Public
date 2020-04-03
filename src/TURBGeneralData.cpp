#include "TURBGeneralData.h"

//TURBGeneralData::TURBGeneralData(Args* arguments, URBGeneralData* UGD){
TURBGeneralData::TURBGeneralData(URBGeneralData* UGD){
    
    auto StartTime = std::chrono::high_resolution_clock::now();
    
    // make local copy of grid information
    // nx,ny,nz consitant with URB (face-center)
    // urb->grid correspond to face-center grid
    nz= UGD->nz;
    ny= UGD->ny;
    nx= UGD->nx;

    float dz= UGD->dz;
    float dy= UGD->dy;
    float dx= UGD->dx;

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
    x_cc = UGD->x;
    // x face-center (this assume constant dx for the moment, same as QES-winds)
    for(int i=1;i<nx-1;i++) {
        x_fc[i]= 0.5*(UGD->x[i-1]+UGD->x[i]);
    }
    x_fc[0] = x_fc[1]-dx;
    x_fc[nx-1] = x_fc[nx-2]+dx;

    // y cell-center
    y_cc = UGD->y;
    // y face-center (this assume constant dy for the moment, same as QES-winds)
    for(int i=1;i<ny-1;i++) {
        y_fc[i] = 0.5*(UGD->y[i-1]+UGD->y[i]);
    }
    y_fc[0] = y_fc[1]-dy;
    y_fc[ny-1] = y_fc[ny-2]+dy;
  
    // z cell-center
    z_cc = UGD->z;
    // z face-center (with ghost cell under the ground)
    for(int i=1;i<nz;i++) {
        z_fc[i] = UGD->z_face[i-1];
    }
    z_fc[0] = z_fc[1]-dz;
    
    // unused: int np_fc = nz*ny*nx;
    int np_cc = (nz-1)*(ny-1)*(nx-1);
  
    iturbflag.resize(np_cc,0);

    /* 
       vector containing cell id of fluid cell
       do not include 1 cell shell around the domain
       => i=1...nx-2 j=1...ny-2
       do not include 1 cell layer at the top of the domain
       => k=0...nz-2 
    */
    for(int k=0;k<nz-2;k++) {
        for(int j=1;j<ny-2;j++) {
            for(int i=1;i<nx-2;i++) {
                int id = i + j*(nx-1) + k*(nx-1)*(ny-1);
                if(UGD->icellflag[id] != 0 && UGD->icellflag[id] != 2) {
                    icellfluid.push_back(id);
                    iturbflag.at(id)=1;
                }
            }
        }
    }
  
  
    // definition of the solid wall for loglaw
    std::cout << "[TURB] \t\t Defining Solid Walls...\n";
    wallVec.push_back(new TURBWallBuilding());
    wallVec.push_back(new TURBWallTerrain());
    /// Boundary condition at wall
    for(auto i=0u;i<wallVec.size();i++) {
        wallVec.at(i)->defineWalls(UGD,this);
    }
    std::cout << "[TURB] \t\t Walls Defined...\n";
  
    // mixing length
    Lm.resize(np_cc,0.0);
    // make a copy as mixing length will be modifiy by non local 
    // (need to be reset at each time instances)
    std::cout << "[TURB] \t\t Defining Local Mixing Length...\n";
    for(auto id=0u;id<icellfluid.size();id++) {
        int idcc=icellfluid[id];
        Lm[idcc]=vonKar*UGD->mixingLengths[idcc];
    }
  
    // comp. of the strain rate tensor
    S11.resize(np_cc,0);
    S12.resize(np_cc,0);
    S13.resize(np_cc,0);
    S22.resize(np_cc,0);
    S23.resize(np_cc,0);
    S33.resize(np_cc,0);
  
    // comp of the stress tensor
    tau11.resize(np_cc,0);
    tau12.resize(np_cc,0);
    tau13.resize(np_cc,0);
    tau22.resize(np_cc,0);
    tau23.resize(np_cc,0);
    tau33.resize(np_cc,0);
  
    // derived turbulence quantities
    tke.resize(np_cc,0);
    CoEps.resize(np_cc,0);

    auto EndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> Elapsed = EndTime - StartTime;
    std::cout << "[TURB] \t\t Memory allocation complete...\n";
    std::cout << "\t\t elapsed time: " << Elapsed.count() << " s" << endl;
    
}

// compute turbulence fields
void TURBGeneralData::run(URBGeneralData* UGD){
    
    auto StartTime = std::chrono::high_resolution_clock::now();
    
    std::cout<<"[TURB] \t\t Computing Derivatives (Strain Rate)"<<std::endl;
    getDerivatives(UGD);
    std::cout<<"[TURB] \t\t Derivatives computed..."<<std::endl;

    std::cout<<"[TURB] \t\t Imposing Wall BC (log law)"<<std::endl;
    for(auto i=0u;i<wallVec.size();i++) {
        wallVec.at(i)->setWallsBC(UGD,this);
    }
    std::cout<<"[TURB] \t\t Wall BC done..."<<std::endl;

    std::cout<<"[TURB] \t\t Computing Stess Tensor"<<std::endl;
    getStressTensor();
    std::cout<<"[TURB] \t\t Stress Tensor computed..."<<std::endl;

    std::cout << "[TURB] \t\t Applying non local mixing..."<<std::endl;;
    for (size_t i = 0; i < UGD->allBuildingsV.size(); i++)
    {
        UGD->allBuildingsV[UGD->building_id[i]]->NonLocalMixing(UGD, this, UGD->building_id[i]);
    }
    
    auto EndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> Elapsed = EndTime - StartTime;
    std::cout << "[TURB] \t\t Turbulence model complete...\n";
    std::cout << "\t\t elapsed time: " << Elapsed.count() << " s" << endl;

    
}


void TURBGeneralData::getDerivatives(URBGeneralData* UGD)
{

    for(auto id=0u;id<icellfluid.size();id++) {
        int id_cc=icellfluid[id];
        //linearized index: id_cc = i + j*(nx-1) + k*(nx-1)*(ny-1);
        // i,j,k -> inverted linearized index
        int k = (int)(id_cc / ((nx-1)*(ny-1)));
        int j = (int)((id_cc - k*(nx-1)*(ny-1))/(nx-1));
        int i = id_cc -  j*(nx-1) - k*(nx-1)*(ny-1);

        /*
          Diagonal componants of the strain-rate tensor naturally fall at
          the cell-center
        */

        // index of neighbour cells
        int id_fc = i + j*nx + k*nx*ny;
        int idxp = id_fc+1;     //i+1,j,k
        int idyp = id_fc+nx;    //i,j+1,k
        int idzp = id_fc+ny*nx; //i,j,k+1

        //S11 = dudx
        S11[id_cc] = (UGD->u[idxp]-UGD->u[id_fc])/(x_fc[i+1]-x_fc[i]);
        //S22 = dvdy
        S22[id_cc] = (UGD->v[idyp]-UGD->v[id_fc])/(y_fc[j+1]-y_fc[j]);
        //S33 = dwdz
        S33[id_cc] = (UGD->w[idzp]-UGD->w[id_fc])/(z_fc[k+1]-z_fc[k]);

        /*
          Off-diagonal componants of the strain-rate tensor require extra interpolation
          of the velocity field to get the derivative at the cell-center
        */

        // index of neighbour cells
        int idp,idm;
        // interpolated velocity field at neighbour cell center
        float up,um,vp,vm,wp,wm;

        //--------------------------------------
        //S12 = 0.5*(dudy+dvdx) at z_cc
        // u_hat+
        idp = id_fc+1+nx; //i+1,j+1
        idm = id_fc+nx;   //i,j+1
        up=((x_cc[i]-x_fc[i])*UGD->u[idp]+(x_fc[i+1]-x_cc[i])*UGD->u[idm])/(x_fc[i+1]-x_fc[i]);

        // u_hat-
        idp = id_fc+1-nx; //i+1,j-1
        idm = id_fc-nx;   //i,j-1
        um=((x_cc[i]-x_fc[i])*UGD->u[idp]+(x_fc[i+1]-x_cc[i])*UGD->u[idm])/(x_fc[i+1]-x_fc[i]);

        // v_hat+
        idp = id_fc+1+nx; //i+1,j+1
        idm = id_fc+1;   //i+1,j
        vp=((y_cc[j]-y_fc[j])*UGD->v[idp]+(y_fc[j+1]-y_cc[j])*UGD->v[idm])/(y_fc[j+1]-y_fc[j]);

        // v_hat-
        idp = id_fc-1+nx; //i-1,j+1
        idm = id_fc-1;   //i-1,j
        vm=((y_cc[j]-y_fc[j])*UGD->v[idp]+(y_fc[j+1]-y_cc[j])*UGD->v[idm])/(y_fc[j+1]-y_fc[j]);

        //S12 = 0.5*(dudy+dvdx) at z_cc
        S12[id_cc] = 0.5*((up-um)/(y_cc[j+1]-y_cc[j-1])+(vp-vm)/(x_cc[i+1]-x_cc[i-1]));

        //--------------------------------------
        //S13 = 0.5*(dudz+dwdx) at y_cc
        // u_hat+
        idp = id_fc+1+nx*ny; //i+1,k+1
        idm = id_fc+nx*ny;   //i,k+1
        up=((x_cc[i]-x_fc[i])*UGD->u[idp]+(x_fc[i+1]-x_cc[i])*UGD->u[idm])/(x_fc[i+1]-x_fc[i]);

        // u_hat-
        idp = id_fc+1-nx*ny; //i+1,k-1
        idm = id_fc-nx*ny;   //i,k-1
        um=((x_cc[i]-x_fc[i])*UGD->u[idp]+(x_fc[i+1]-x_cc[i])*UGD->u[idm])/(x_fc[i+1]-x_fc[i]);

        // w_hat+
        idp = id_fc+1+nx*ny; //i+1,k+1
        idm = id_fc+1;   //i+1,k
        wp=((z_cc[k]-z_fc[k])*UGD->w[idp]+(z_fc[k+1]-z_cc[k])*UGD->w[idm])/(z_fc[k+1]-z_fc[k]);

        // w_hat-
        idp = id_fc-1+nx*ny; //i-1,k+1
        idm = id_fc-1;   //i-1,k
        wm=((z_cc[k]-z_fc[k])*UGD->w[idp]+(z_fc[k+1]-z_cc[k])*UGD->w[idm])/(z_fc[k+1]-z_fc[k]);

        //S13 = 0.5*(dudz+dwdx) at y_cc
        S13[id_cc] = 0.5*((up-um)/(z_cc[k+1]-z_cc[k-1])+(wp-wm)/(x_cc[i+1]-x_cc[i-1]));

        //--------------------------------------
        //S23 = 0.5*(dvdz+dwdy) at x_cc
        // v_hat+
        idp = id_fc+nx+nx*ny; //j+1,k+1
        idm = id_fc+nx*ny;   //j,k+1
        vp=((y_cc[j]-y_fc[j])*UGD->v[idp]+(y_fc[j+1]-y_cc[j])*UGD->v[idm])/(y_fc[j+1]-y_fc[j]);

        // v_hat-
        idp = id_fc+nx-nx*ny; //j+1,k-1
        idm = id_fc-nx*ny;   //j,k-1
        vm=((y_cc[j]-y_fc[j])*UGD->v[idp]+(y_fc[j+1]-y_cc[j])*UGD->v[idm])/(y_fc[j+1]-y_fc[j]);

        // w_hat+
        idp = id_fc+nx+nx*ny; //j+1,k+1
        idm = id_fc+nx;   //j+1,k
        wp=((z_cc[k-1]-z_fc[k])*UGD->w[idp]+(z_fc[k+1]-z_cc[k])*UGD->w[idm])/(z_fc[k+1]-z_fc[k]);

        // w_hat-
        idp = id_fc-nx+nx*ny; //j-1,k+1
        idm = id_fc-nx;   //j-1,k
        wp=((z_cc[k]-z_fc[k])*UGD->w[idp]+(z_fc[k+1]-z_cc[k])*UGD->w[idm])/(z_fc[k+1]-z_fc[k]);

        //S23 = 0.5*(dvdz+dwdy) at x_cc
        S23[id_cc] = 0.5*((vp-vm)/(z_cc[k+1]-z_cc[k-1])+(wp-wm)/(y_cc[j+1]-y_cc[j-1]));

    }
}

void TURBGeneralData::getStressTensor()
{
    int id_cc;
    for(auto id=0u;id<icellfluid.size();id++) {
        id_cc=icellfluid[id];

        float NU_T = 0.0;
        float TKE  = 0.0;
        float LM = Lm[id_cc];

        //
        float SijSij = S11[id_cc]*S11[id_cc] + S22[id_cc]*S22[id_cc] + S33[id_cc]*S33[id_cc]
            + 2.0*(S12[id_cc]*S12[id_cc] + S13[id_cc]*S13[id_cc] + S23[id_cc]*S23[id_cc]);

        NU_T = LM*LM*sqrt(2.0*SijSij);
        TKE  = pow((NU_T/(cPope*LM)),2.0);
        tke[id_cc] = TKE;
        CoEps[id_cc] = 5.7*pow(sqrt(TKE)*cPope,3.0)/(LM);

        tau11[id_cc] = (2.0/3.0) * TKE - 2.0*(NU_T*S11[id_cc]);
        tau22[id_cc] = (2.0/3.0) * TKE - 2.0*(NU_T*S22[id_cc]);
        tau33[id_cc] = (2.0/3.0) * TKE - 2.0*(NU_T*S33[id_cc]);
        tau12[id_cc] = - 2.0*(NU_T*S12[id_cc]);
        tau13[id_cc] = - 2.0*(NU_T*S13[id_cc]);
        tau23[id_cc] = - 2.0*(NU_T*S23[id_cc]);

        tau11[id_cc] = fabs(sigUConst*tau11[id_cc]);
        tau22[id_cc] = fabs(sigVConst*tau22[id_cc]);
        tau33[id_cc] = fabs(sigWConst*tau33[id_cc]);

    }
}

