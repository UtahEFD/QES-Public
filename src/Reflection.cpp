#include "Plume.hpp"

bool Plume::reflection(URBGeneralData* UGD, Eulerian* eul,
                       double& xPos, double& yPos, double& zPos, 
                       double& disX, double& disY, double& disZ,
                       double& uFluct, double& vFluct, double& wFluct)
{
    /*
      This function will return false and leave xPos, yPos, zPos, uFluct, vFluct, wFluct unchanged
      if: 
       - UGD->icellflag.at(cellIdNew) == 0 || UGD->icellflag.at(cellIdNew) == 2
       - count > maxCount
       - cell ID out of bound
       - particle trajectory more than 1 cell in each direction
    */

    // some constants
    const double eps_S = 0.0001;
    const int maxCount = 10;
    
    // QES-winds grid information
    int nx = UGD->nx;
    int ny = UGD->ny;
    int nz = UGD->nz;
    double dx = UGD->dx;
    double dy = UGD->dy;
    double dz = UGD->dz;

    // linearized cell ID for origine and end of the trajectory of the particle
    int cellIdOld,cellIdNew;
    // icellFlag of the cell at the end of the trajectory of the particle 
    int cellFlagNew;
    
    // cartesian basis vectors 
    Vector3<double> e1={1.0,0.0,0.0},e2={0.0,1.0,0.0},e3={0.0,0.0,1.0};
    
    /* 
       Vector3 variables informations:
       Xold     = origine of the trajectory of the particle  
       Xnew     = end of the trajectory of the particle 
       vecFluct = fluctuation of the particle
       P        = position of the particle on the wall where bounce happens
       S        = location of the wall
       U        = trajectory of the particle
       V1       = trajectory of the particle to the wall V1 = P - Xold
       V2       = trajectory of the particle to the wall V1 = Xnew - P
       R        = unit vector giving orentation of the reflection
       N        = unit vector noraml to the surface
    */
    Vector3<double> Xnew,Xold;
    Vector3<double> vecFluct;    
    Vector3<double> P,S,U,V1,V2;
    Vector3<double> R,N;

    // position of the particle start of trajectory
    Xold={xPos-disX,yPos-disY,zPos-disZ};

    // postion of the particle end of trajectory 
    Xnew={xPos,yPos,zPos};

    // cell ID of the origin and end of the particle trajectory  
    cellIdOld=eul->getCellId(Xold);
    cellIdNew=eul->getCellId(Xnew); 
    
    Vector3<int> cellIdxOld=eul->getCellIndex(cellIdOld);
    Vector3<int> cellIdxNew=eul->getCellIndex(cellIdNew);
    
    // check particle trajectory more than 1 cell in each direction
    if( (abs(cellIdxOld[0]-cellIdxNew[0]) > 1) ||
        (abs(cellIdxOld[1]-cellIdxNew[1]) > 1) ||
        (abs(cellIdxOld[0]-cellIdxNew[0]) > 1) ) {
        //std::cout << "particle trajectory more than 1 cell in each direction" << std::enld;
        return false;
    }

    int i=cellIdxOld[0], j=cellIdxOld[1], k=cellIdxOld[2];

    // cell type of the end point
    cellFlagNew=UGD->icellflag.at(cellIdNew);

    // vector of fluctuation
    vecFluct={uFluct,vFluct,wFluct};

    /* 
       Working variables informations:
       count       - number of reflections
       f1,f2,f3    - sign of trajectory in each direction (+/-1)
       l1,l2,l3    - ratio of distance to wall over total distance travel to closest surface in 
       -             each direction: by definition positive, if < 1 -> reflection possible
       -             if > 1 -> surface too far
       validSuface - number of potential valid surface
       s           - smallest ratio of dist. to wall over total dist. travel (once surface selected)
       d           - distance travel after reflection
    */
    int count=0;
    double f1,f2,f3;
    double l1,l2,l3;
    int validSurface;
    double s,d;
    
    while( (cellFlagNew==0 || cellFlagNew==2) && (count < maxCount) ){ 
        
        // distance travelled by particle 
        U=Xnew-Xold;
        
        //set direction 
        f1=(e1*U);f1=f1/std::abs(f1);
        f2=(e2*U);f2=f2/std::abs(f2);
        f3=(e3*U);f3=f3/std::abs(f3);
        
        // reset smallest ratio
        s=100.0;
        // reset number of potential valid surface
        validSurface=0;
        // reset distance travel after all
        d=0.0;
                
        // x-drection
        N=-f1*e1;
        S={UGD->x[i]+f1*0.5*dx,double(UGD->y[j]),double(UGD->z[k])};
        l1 = - (Xold*N - S*N)/(U*N);

        // y-drection
        N=-f2*e2;
        S={double(UGD->x[i]),UGD->y[j]+f2*0.5*dy,double(UGD->z[k])};
        l2 = -(Xold*N - S*N)/(U*N);
        
        // z-drection (dz can be variable with hieght)
        N=-f3*e3;
        if(f3 >= 0.0) {
            S={double(UGD->x[i]),double(UGD->y[j]),double(UGD->z_face[k])};
        } else {
            S={double(UGD->x[i]),double(UGD->y[j]),double(UGD->z_face[k-1])};
        }
        l3 = -(Xold*N - S*N)/(U*N);
        
        // check with surface is a potential bounce (li < 1.0)
        if( (l1 < 1.0) && (l1 >= -eps_S) ){
            validSurface ++;
            s=l1;
            N=-f1*e1;
        }
        if( (l2 < 1.0) && (l2 >= -eps_S) ){
            validSurface ++;
            s=l2;
            N=-f2*e2;
        }
        if( (l3 < 1.0) && (l3 >= -eps_S) ){
            validSurface ++;
            s=l3;
            N=-f3*e3;
        }
        
        // check if more than one surface is valid
        if(validSurface == 0) {
            // if 0 valid surface  
            //std::cout << "Reflection problem: no valid surface" << std::endl;
            return false;
        } else if(validSurface > 1) {
            // Here-> Multiple options to bounce
            // NOTE: vectors will be at min size 2!
            
            // list of potential surface (will be sorted from smallest to largest)
            // - ratio of dist. to wall over dist. total
            std::vector<double> vl;
            // - normal vector for each surface
            std::vector<Vector3<double>> vN;
            // - linear index for icellflag check
            std::vector<int> vn;
            
            // add potential surface to the list only if (li < 1.0)
            if (l1 < 1.0) {
                vl.push_back(l1);
                vN.push_back(-f1*e1);
                vn.push_back(f1);
            }
            if (l2 < 1.0) {
                vl.push_back(l2);
                vN.push_back(-f2*e2);
                vn.push_back(f2*(nx-1));
            }
            if (l3 < 1.0) {
                vl.push_back(l3);
                vN.push_back(-f3*e3);
                vn.push_back(f3*(nx-1)*(ny-1));
            }
            
            // sort from smallest to largest and retain index 
            std::vector<size_t> idx(vl.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&vl](size_t i1, size_t i2) {return vl[i1] < vl[i2];});
            
            // check if surface is valid (ie, next cell is solid)
            if( (UGD->icellflag.at(cellIdOld+vn[idx[0]]) == 0) || 
                (UGD->icellflag.at(cellIdOld+vn[idx[0]]) == 2) ) {
                s=vl[idx[0]];
                N=vN[idx[0]];
            } else if( (UGD->icellflag.at(cellIdOld+vn[idx[0]]+vn[idx[1]]) == 0) || 
                       (UGD->icellflag.at(cellIdOld+vn[idx[0]]+vn[idx[1]]) == 2) ) {
                s=vl[idx[1]];
                N=vN[idx[1]];
            } else if (idx.size() == 3) { // check if 3rd option is valid (avoid seg fault)
                if( (UGD->icellflag.at(cellIdOld+vn[idx[0]]+vn[idx[1]]+vn[idx[2]]) == 0) || 
                    (UGD->icellflag.at(cellIdOld+vn[idx[0]]+vn[idx[1]]+vn[idx[2]]) == 2) ) {
                    s=vl[idx[2]];
                    N=vN[idx[2]];
                } else {
                    // this should happend only if particle traj. more than 1 cell in each direction,
                    // -> should have been skipped at the beginning of the function
                    //std::cout << "Reflection problem: no valid surface" << std::endl;
                    return false;
                } 
            }
        }
        // no else -> only one valid surface
        
                
        // vector from current postition to the wall
        V1=s*U;
        // postion of relfection on the wall
        P=Xold+V1;
        // distance traveled after the wall
        V2=U-V1;
        d=V2.length();
        // reflection: normalizing V2 -> R is of norm 1
        V2=V2/V2.length();
        R=V2.reflect(N);
        // update postion from surface reflection
        Xnew=P+d*R;
        
        // relfection of the Fluctuation 
        vecFluct=vecFluct.reflect(N);
        
        // prepare variables for next bounce: particle position
        Xold=P;

        // increment the relfection count
        count=count+1;                
        // update the icellflag
        cellIdNew=eul->getCellId(Xnew);
        
        try {
            cellFlagNew=UGD->icellflag.at(cellIdNew);
        } catch (const std::out_of_range& oor) {            
            // cell ID out of bound
            std::cout << "Reflection problem: particle out of range" << std::endl;
            return false;
        }
        
    } // end of: while( (cellFlagNew==0 || cellFlagNew==2) && (count < maxCount) ) 
    
    // if count run out -> function return false
    if(count >= maxCount){
        return false;
    } else {
        // update output variable: particle position
        xPos=Xnew[0];
        yPos=Xnew[1];
        zPos=Xnew[2];
        // update output variable: fluctuations
        uFluct=vecFluct[0];
        vFluct=vecFluct[1];
        wFluct=vecFluct[2];    
        
        return true;
    }
    
    /*
      This function will return false and leave xPos, yPos, zPos, uFluct, vFluct, wFluct unchanged
      if: 
       - UGD->icellflag.at(cellIdNew) == 0 || UGD->icellflag.at(cellIdNew) == 2
       - count > maxCount
       - cell ID out of bound
       - particle trajectory more than 1 cell in each direction
    */
    
}

