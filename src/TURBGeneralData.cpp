#include "TURBGeneralData.h"

//TURBGeneralData::TURBGeneralData(Args* arguments, URBGeneralData* WGD){
TURBGeneralData::TURBGeneralData(const WINDSInputData* WID,WINDSGeneralData* WGD){

    auto StartTime = std::chrono::high_resolution_clock::now();

    std::cout << "[QES-TURB]\t Initialization of turbulence model...\n";

    // make local copy of grid information
    // nx,ny,nz consitant with WINDS (face-center)
    // WINDS->grid correspond to face-center grid
    nz= WGD->nz;
    ny= WGD->ny;
    nx= WGD->nx;

    float dz= WGD->dz;
    float dy= WGD->dy;
    float dx= WGD->dx;

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
                if(WGD->icellflag[id] != 0 && WGD->icellflag[id] != 2) {
                    icellfluid.push_back(id);
                    iturbflag.at(id)=1;
                }
            }
        }
    }


    // definition of the solid wall for loglaw
    std::cout << "\t\t Defining Solid Walls...\n";
    wallVec.push_back(new TURBWallBuilding());
    wallVec.push_back(new TURBWallTerrain());
    /// Boundary condition at wall
    for(auto i=0u;i<wallVec.size();i++) {
        wallVec.at(i)->defineWalls(WGD,this);
    }
    //std::cout << "\t\t Walls Defined...\n";

    // mixing length
    Lm.resize(np_cc,0.0);
    // make a copy as mixing length will be modifiy by non local
    // (need to be reset at each time instances)
    std::cout << "\t\t Defining Local Mixing Length...\n";
    for(auto id=0u;id<icellfluid.size();id++) {
        int idcc=icellfluid[id];
        Lm[idcc]=vonKar*WGD->mixingLengths[idcc];
    }
    
    // Caluclating Turbulence quantities (u*,z0,d0) based on morphology of domain.
    bldgH_mean=0.0;
    bldgH_max=0.0;
    terrainH_max=0.0;
    zRef=0.0;uRef=0.0;uStar=0.0;

    std::cout << "\t\t Calculating Morphometric parametrization of trubulence..."<<std::endl;
    
    if (WID->simParams->DTE_heightField) {
        terrainH_max=*max_element(WGD->terrain.begin(), WGD->terrain.end());
    } else {
        terrainH_max=0.0;
    }

    //std::cout << "\t\t max terrain height = "<< terrainH_max << std::endl;

    // calculate the mean building h
    if(WGD->allBuildingsV.size() > 0) {
        float heffmax = 0.0;
        for (size_t i = 0; i < WGD->allBuildingsV.size(); i++) {
            bldgH_mean += WGD->allBuildingsV[WGD->building_id[i]]->H;
            heffmax=WGD->allBuildingsV[WGD->building_id[i]]->H;//height_eff;
            if(heffmax > bldgH_max) {
                bldgH_max = heffmax;
            }
        }
        bldgH_mean = bldgH_mean/float(WGD->allBuildingsV.size());
        
        //std::cout << "\t\t\t mean bldg height = "<< bldgH_mean << " max bldg height = "<< bldgH_max << std::endl;
    
        // Morphometric parametrization based on Grimmond and Oke (1999) and Kaster-Klein and Rotach (2003)
        // roughness length z0 as 0.1 mean building height
        z0d = 0.1*bldgH_mean;
        // displacement height d0 as 0.7 mean building height
        d0d = 0.7*bldgH_mean;

        // reference height as 3.0 mean building 
        zRef=3.0*bldgH_mean;
        
    } else {
        z0d = WID->metParams->sensors[0]->TS[0]->site_z0;
        d0d = 0.0;
        zRef =100.0*z0d;
    }

    std::cout<<"\t\t Computing friction velocity..." << std::endl;
    getFrictionVelocity(WGD);
        

    std::cout << "\t\t Allocating memory...\n";

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
    //std::cout << "\t\t Memory allocation completed.\n";

    auto EndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> Elapsed = EndTime - StartTime;

    std::cout << "[QES-TURB]\t Initialization of turbulence model completed.\n";
    std::cout << "\t\t elapsed time: " << Elapsed.count() << " s" << endl;

}

// compute turbulence fields
void TURBGeneralData::run(WINDSGeneralData* WGD){

    auto StartTime = std::chrono::high_resolution_clock::now();

    std::cout<<"[QES-TURB] \t Running turbulence model..."<<std::endl;

    std::cout<<"\t\t Computing friction velocity..." << std::endl;
    getFrictionVelocity(WGD);

    std::cout<<"\t\t Computing Derivatives (Strain Rate)..."<<std::endl;
    getDerivatives(WGD);
    //std::cout<<"\t\t Derivatives computed."<<std::endl;

    std::cout<<"\t\t Imposing Wall BC (log law)..."<<std::endl;
    for(auto i=0u;i<wallVec.size();i++) {
        wallVec.at(i)->setWallsBC(WGD,this);
    }
    std::cout<<"\t\t Wall BC done."<<std::endl;

    std::cout<<"\t\t Computing Stess Tensor..."<<std::endl;
    getStressTensor();
    //std::cout<<"\t\t Stress Tensor computed."<<std::endl;

    std::cout<<"\t\t Applying non-local mixing..."<<std::endl;;
    for (size_t i = 0; i < WGD->allBuildingsV.size(); i++) {
        WGD->allBuildingsV[WGD->building_id[i]]->NonLocalMixing(WGD, this, WGD->building_id[i]);
    }
    //std::cout<<"\t\t Non-local mixing completed."<<std::endl;

    std::cout<<"\t\t Capping Stess Tensor..."<<std::endl;
    capStressTensor();

    auto EndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> Elapsed = EndTime - StartTime;
    
    std::cout << "[QES-TURB] \t Turbulence model completed.\n";
    std::cout << "\t\t elapsed time: " << Elapsed.count() << " s" << endl;


}

void TURBGeneralData::getFrictionVelocity(WINDSGeneralData* WGD) 
{
    float nVal=0.0,uSum=0.0;
    for (int j = 0; j < ny-1; j++) {
        for (int i = 0; i < nx-1; i++) {
            // search the vector for the first element with value 42
            std::vector<float>::iterator itr = std::lower_bound(z_fc.begin(), z_fc.end(), zRef);
            int k;
            if (itr != z_fc.end()) {
                k=itr - z_fc.begin();
                //std::cout << "\t\t\t ref height = "<< zRef << " kRef = "<< kRef << std::endl;
            }else{
                std::cout << "\t\t\t DOMAIN TOO SMALL " << std::endl;   
                k=1;
            }
            
            int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
            int icell_face = i + j*(nx) + k*(nx)*(ny);
            if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                uSum += sqrt(pow(0.5*(WGD->u[icell_face]+WGD->u[icell_face+1]),2) + 
                             pow(0.5*(WGD->v[icell_face]+WGD->v[icell_face+nx]),2) +
                             pow(0.5*(WGD->w[icell_face]+WGD->w[icell_face+nx*ny]),2));
                nVal++;
            }
        }
    }
    uRef=uSum/nVal;
    
    uStar = 0.4*uRef/log((zRef-d0d)/z0d);
    std::cout << "\t\t Mean reference velocity uRef = " << uRef << " m/s" << std::endl;
    std::cout << "\t\t Mean friction velocity uStar = " << uStar << " m/s" << std::endl;

}

void TURBGeneralData::getDerivatives(WINDSGeneralData* WGD)
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
        S11[id_cc] = (WGD->u[idxp]-WGD->u[id_fc])/(x_fc[i+1]-x_fc[i]);
        //S22 = dvdy
        S22[id_cc] = (WGD->v[idyp]-WGD->v[id_fc])/(y_fc[j+1]-y_fc[j]);
        //S33 = dwdz
        S33[id_cc] = (WGD->w[idzp]-WGD->w[id_fc])/(z_fc[k+1]-z_fc[k]);

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
        up=((x_cc[i]-x_fc[i])*WGD->u[idp]+(x_fc[i+1]-x_cc[i])*WGD->u[idm])/(x_fc[i+1]-x_fc[i]);

        // u_hat-
        idp = id_fc+1-nx; //i+1,j-1
        idm = id_fc-nx;   //i,j-1
        um=((x_cc[i]-x_fc[i])*WGD->u[idp]+(x_fc[i+1]-x_cc[i])*WGD->u[idm])/(x_fc[i+1]-x_fc[i]);

        // v_hat+
        idp = id_fc+1+nx; //i+1,j+1
        idm = id_fc+1;   //i+1,j
        vp=((y_cc[j]-y_fc[j])*WGD->v[idp]+(y_fc[j+1]-y_cc[j])*WGD->v[idm])/(y_fc[j+1]-y_fc[j]);

        // v_hat-
        idp = id_fc-1+nx; //i-1,j+1
        idm = id_fc-1;   //i-1,j
        vm=((y_cc[j]-y_fc[j])*WGD->v[idp]+(y_fc[j+1]-y_cc[j])*WGD->v[idm])/(y_fc[j+1]-y_fc[j]);

        //S12 = 0.5*(dudy+dvdx) at z_cc
        S12[id_cc] = 0.5*((up-um)/(y_cc[j+1]-y_cc[j-1])+(vp-vm)/(x_cc[i+1]-x_cc[i-1]));

        //--------------------------------------
        //S13 = 0.5*(dudz+dwdx) at y_cc
        // u_hat+
        idp = id_fc+1+nx*ny; //i+1,k+1
        idm = id_fc+nx*ny;   //i,k+1
        up=((x_cc[i]-x_fc[i])*WGD->u[idp]+(x_fc[i+1]-x_cc[i])*WGD->u[idm])/(x_fc[i+1]-x_fc[i]);

        // u_hat-
        idp = id_fc+1-nx*ny; //i+1,k-1
        idm = id_fc-nx*ny;   //i,k-1
        um=((x_cc[i]-x_fc[i])*WGD->u[idp]+(x_fc[i+1]-x_cc[i])*WGD->u[idm])/(x_fc[i+1]-x_fc[i]);

        // w_hat+
        idp = id_fc+1+nx*ny; //i+1,k+1
        idm = id_fc+1;   //i+1,k
        wp=((z_cc[k]-z_fc[k])*WGD->w[idp]+(z_fc[k+1]-z_cc[k])*WGD->w[idm])/(z_fc[k+1]-z_fc[k]);

        // w_hat-
        idp = id_fc-1+nx*ny; //i-1,k+1
        idm = id_fc-1;   //i-1,k
        wm=((z_cc[k]-z_fc[k])*WGD->w[idp]+(z_fc[k+1]-z_cc[k])*WGD->w[idm])/(z_fc[k+1]-z_fc[k]);

        //S13 = 0.5*(dudz+dwdx) at y_cc
        S13[id_cc] = 0.5*((up-um)/(z_cc[k+1]-z_cc[k-1])+(wp-wm)/(x_cc[i+1]-x_cc[i-1]));

        //--------------------------------------
        //S23 = 0.5*(dvdz+dwdy) at x_cc
        // v_hat+
        idp = id_fc+nx+nx*ny; //j+1,k+1
        idm = id_fc+nx*ny;   //j,k+1
        vp=((y_cc[j]-y_fc[j])*WGD->v[idp]+(y_fc[j+1]-y_cc[j])*WGD->v[idm])/(y_fc[j+1]-y_fc[j]);

        // v_hat-
        idp = id_fc+nx-nx*ny; //j+1,k-1
        idm = id_fc-nx*ny;   //j,k-1
        vm=((y_cc[j]-y_fc[j])*WGD->v[idp]+(y_fc[j+1]-y_cc[j])*WGD->v[idm])/(y_fc[j+1]-y_fc[j]);

        // w_hat+
        idp = id_fc+nx+nx*ny; //j+1,k+1
        idm = id_fc+nx;   //j+1,k
        wp=((z_cc[k-1]-z_fc[k])*WGD->w[idp]+(z_fc[k+1]-z_cc[k])*WGD->w[idm])/(z_fc[k+1]-z_fc[k]);

        // w_hat-
        idp = id_fc-nx+nx*ny; //j-1,k+1
        idm = id_fc-nx;   //j-1,k
        wp=((z_cc[k]-z_fc[k])*WGD->w[idp]+(z_fc[k+1]-z_cc[k])*WGD->w[idm])/(z_fc[k+1]-z_fc[k]);

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


void TURBGeneralData::capStressTensor()
{
    int id_cc;
    float stressCap = 100*uStar*uStar;
    for(auto id=0u;id<icellfluid.size();id++) {
        id_cc=icellfluid[id];

        if (tau11[id_cc] < -stressCap)
            tau11[id_cc] = -stressCap;
        if (tau11[id_cc] > stressCap) 
            tau11[id_cc] = stressCap;
        
        if (tau12[id_cc] < -stressCap)
            tau12[id_cc] = -stressCap;
        if (tau12[id_cc] > stressCap) 
            tau12[id_cc] = stressCap;
        
        if (tau13[id_cc] < -stressCap) 
            tau13[id_cc] = -stressCap;
        if (tau13[id_cc] > stressCap) 
            tau13[id_cc] = stressCap;

        if (tau22[id_cc] < -stressCap)
            tau22[id_cc] = -stressCap;
        if (tau22[id_cc] > stressCap) 
            tau22[id_cc] = stressCap;

        if (tau23[id_cc] < -stressCap)
            tau23[id_cc] = -stressCap;
        if (tau23[id_cc] > stressCap) 
            tau23[id_cc] = stressCap;
        
        if (tau33[id_cc] < -stressCap) 
            tau33[id_cc] = -stressCap;
        if (tau33[id_cc] > stressCap)
            tau33[id_cc] = stressCap;
    }
}

