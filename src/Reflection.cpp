#include "Plume.hpp"

bool Plume::reflection(URBGeneralData* UGD, Eulerian* eul,
                       double& xPos, double& yPos, double& zPos, 
                       double& disX, double& disY, double& disZ,
                       double& uFluct, double& vFluct, double& wFluct)
{
    
    const double eps_S = 0.0001;
    const int maxCount = 10;
    double smallestS;
    
    int nx = UGD->nx;
    int ny = UGD->ny;
    int nz = UGD->nz;

    double dx = UGD->dx;
    double dy = UGD->dy;
    double dz = UGD->dz;

    int cellIdOld,cellIdNew,cellFlagNew;
    
    Vector3<double> e1={1.0,0.0,0.0},e2={0.0,1.0,0.0},e3={0.0,0.0,1.0};
    
    Vector3<double> vecFluct;
    Vector3<double> Xnew,Xold,P,S;
    Vector3<double> U,V1,V2;
    Vector3<double> R,N;

    // position of the particle start of trajectory
    Xold[0]=xPos - disX;
    Xold[1]=yPos - disY;
    Xold[2]=zPos - disZ;

    // postion of the particle end of trajectory 
    Xnew[0]=xPos;
    Xnew[1]=yPos;
    Xnew[2]=zPos;

        // cell ID of the origin and end of the particle trajectory  
    cellIdOld=eul->getCellId(Xold[0],Xold[1],Xold[2]); 
    cellIdNew=eul->getCellId(Xnew[0],Xnew[1],Xnew[2]); 
    
    // cell type of the end point
    cellFlagNew=UGD->icellflag.at(cellIdNew);

    // vector of fluctuation
    vecFluct[0]=uFluct;
    vecFluct[1]=vFluct;
    vecFluct[2]=wFluct;

    int count=0;
    int i,j,k;
    double f1,f2,f3;
    double l,l1,l2,l3;
    int validSurface;
    std::string bounce;
    double s1,s2,s3,s4,s5,s6;
    double s,d;
    while( (cellFlagNew==0 || cellFlagNew==2 ) && count < maxCount ){ //pos.z<0.0 covers ground reflections
        
        // distance travelled by particle 
        U=Xnew-Xold;
        
        //set direction 
        f1=(e1*U);f1=f1/std::abs(f1);
        f2=(e2*U);f2=f2/std::abs(f2);
        f3=(e3*U);f3=f3/std::abs(f3);
        
        //smallest ratio
        s=100.0;
        validSurface=0;
        //distance travel after all
        d=0.0;
        
        k = (int)(cellIdOld / ((nx-1)*(ny-1)));
        j = (int)((cellIdOld - k*(nx-1)*(ny-1))/(nx-1));
        i = cellIdOld -  j*(nx-1) - k*(nx-1)*(ny-1);
        
        N=-f1*e1;
        S={UGD->x[i]+f1*0.5*dx,double(UGD->y[j]),double(UGD->z[k])};
        l1 = - (Xold*N - S*N)/(U*N);

        N=-f2*e2;
        S={double(UGD->x[i]),UGD->y[j]+f2*0.5*dy,double(UGD->z[k])};
        l2 = -(Xold*N - S*N)/(U*N);
        
        N=-f3*e3;
        if(f3 >= 0.0) {
            S={double(UGD->x[i]),double(UGD->y[j]),double(UGD->z_face[k])};
        } else {
            S={double(UGD->x[i]),double(UGD->y[j]),double(UGD->z_face[k-1])};
        }
        l3 = -(Xold*N - S*N)/(U*N);
        
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
            std::cout << "Reflection problem: no valid surface (test 1)" << std::endl;
            return false;
        } else if(validSurface > 1) {
            //std::cout << "Multiple options" << std::endl;
                        
            std::vector<double> vl;
            std::vector<Vector3<double>> vN;
            std::vector<int> vn;
            
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
            
            std::vector<size_t> idx(vl.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&vl](size_t i1, size_t i2) {return vl[i1] < vl[i2];});
            try {
                if( (UGD->icellflag.at(cellIdOld+vn[idx[0]]) == 0) || 
                    (UGD->icellflag.at(cellIdOld+vn[idx[0]]) == 2) ) {
                    s=vl[idx[0]];
                    N=vN[idx[0]];
                } else if( (UGD->icellflag.at(cellIdOld+vn[idx[0]]+vn[idx[1]]) == 0) || 
                           (UGD->icellflag.at(cellIdOld+vn[idx[0]]+vn[idx[1]]) == 2) ) {
                    s=vl[idx[1]];
                    N=vN[idx[1]];
                } else if( (UGD->icellflag.at(cellIdOld+vn[idx[0]]+vn[idx[1]]+vn[idx[2]]) == 0) || 
                           (UGD->icellflag.at(cellIdOld+vn[idx[0]]+vn[idx[1]]+vn[idx[2]]) == 2) ) {
                    s=vl[idx[2]];
                    N=vN[idx[2]];
                } else {
                    //std::cout << "Reflection problem: no valid surface (test 2)" << std::endl;
                    return false;
                } 
            } catch (const std::out_of_range& oor) {            
                //std::cout << "Reflection problem: no valid surface (catch)" << std::endl;
                return false;

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
        R=reflect(V2,N);
        // update postion from surface reflection
        Xnew=P+d*R;
        
        // relfection of the Fluctuation 
        vecFluct = reflect(vecFluct,N);
        
        // prepare variables for next bounce: particle position
        Xold=P;
        
        // increment the relfection count
        count=count+1;                
        // update the icellflag
        cellIdNew=eul->getCellId(Xnew[0],Xnew[1],Xnew[2]);
        
        try {
            //std::cerr << "particle position after reflection: "<< Xnew << std::endl;
            cellFlagNew=UGD->icellflag.at(cellIdNew);
            //std::cerr << "cellFlagNew: "<< cellFlagNew << std::endl;;
        } catch (const std::out_of_range& oor) {            
            std::cout << i << " " << j << " " << k << std::endl;
            std::cout << l1 << " " << l2 << " " << l3 << " " << s << std::endl;
            std::cout << cellFlagNew << Xnew << Xold << U << std::endl;
            std::cout << "Reflection problem: particle out of range" << std::endl;
            return false;
        }
        
    }
    
    if(UGD->icellflag.at(cellIdNew) == 0 || UGD->icellflag.at(cellIdNew) == 2) {
        //std::cout << "Reflection problem: need more reflection" << std::endl;
        return false;
    }

    // if count run out -> set particle to isActive=false
    if(count > maxCount){
        //std::cout << "Reflection problem: too many reflection" << std::endl;
        return false;
    
    } else {
        
        //std::cout << Xnew << std::endl;
        
        // update output variable
        // particle position
        xPos=Xnew[0];
        yPos=Xnew[1];
        zPos=Xnew[2];
        
        // fluctuation
        uFluct=vecFluct[0];
        vFluct=vecFluct[1];
        wFluct=vecFluct[2];    
        
        return true;
    }
}

