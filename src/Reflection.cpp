#include "Plume.hpp"

bool Plume::reflection(URBGeneralData* UGD, Eulerian* eul,
                       double& xPos, double& yPos, double& zPos, 
                       double& disX, double& disY, double& disZ,
                       double& uFluct, double& vFluct, double& wFluct)
{
    
    const double eps_S = 0.0001;
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
    while( (cellFlagNew==0 || cellFlagNew==2 ) && count<1 ){ //pos.z<0.0 covers ground reflections
        
        // distance travelled by particle 
        U=Xnew-Xold;
        
        //set direction 
        f1=(e1*U);f1=f1/std::abs(f1);
        f2=(e2*U);f2=f2/std::abs(f2);
        f3=(e3*U);f3=f3/std::abs(f3);
                
        //set the ratio of distance travel in each of the 6 directions
        s1 = -1.0; //for x
        s2 = -1.0; //for y
        s3 = -1.0; //for z
        
        s4 = -1.0; //for +y
        s5 = -1.0; //for -z
        s6 = -1.0; //for +z
        
        //smallest ratio
        s=100.0;
        l=100.0;
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
        
        if( (l1 < 1) && (l1 >= -eps_S) ){
            validSurface ++;
            l=l1;
            N=-f1*e1;
        }
        if( (l2 < 1) && (l2 >= -eps_S) ){
            validSurface ++;
            l=l2;
            N=-f2*e2;
        }
        if( (l3 < 1) && (l3 >= -eps_S) ){
            validSurface ++;
            l=l3;
            N=-f3*e3;
        }
        
        if(validSurface > 1) {
            std::cout << "Multiple options" << std::endl;
        } else {
            std::cout << "single option" << std::endl;
        }
        
        //std::cout << f1 << " " << f2 << " " << f3 << std::endl; 
        //std::cout << l1 << " " << l2 << " " << l3 << std::endl; 
        // select which surface the particle is reflecting of:
        if((l1 < l) && (l1 >= -eps_S)){
            l=l1;
            bounce = 'X';
        }
        if((l2 < l) && (l2 >= -eps_S)){
            l=l2;
            bounce = 'Y';
        }
        if((l3 < l) && (l3 >= -eps_S)){
            l=l3;
            bounce = 'Z';
        }
        //std::cout << bounce << std::endl;
        if( (l1 >= 1) && (l2 >= 1) && (l3 >= 1) ) {
            std::cout << "calculation error" << std::endl; 
        }
            

        // -x normal (ckeck cell at i-1)
        if(UGD->icellflag.at(cellIdOld-1) == 0 || UGD->icellflag.at(cellIdOld-1) == 2) {
            N={-1.0,0.0,0.0};    
            S={UGD->x[i]-0.5*dx,0.0,0.0};
            s1 = -(Xold*N - S*N)/(U*N);
        }
        // +x normal (check cell at i+1)
        if(UGD->icellflag.at(cellIdOld+1) == 0 || UGD->icellflag.at(cellIdOld+1) == 2) {
            N={1.0,0.0,0.0};    
            S={UGD->x[i]+0.5*dx,0.0,0.0};
            s2 = -(Xold*N - S*N)/(U*N);
        }
        
        // -y normal (check cell at j-1)
        if(UGD->icellflag.at(cellIdOld-(nx-1)) == 0 || UGD->icellflag.at(cellIdOld-(nx-1)) == 2) {
            N={0.0,-1.0,0.0};
            S={0.0,UGD->y[j]-0.5*dy,0.0}; 
            s3 = -(Xold*N - S*N)/(U*N);
        }
        // +y normal (check cell at j+1)
        if(UGD->icellflag.at(cellIdOld+(nx-1)) == 0 || UGD->icellflag.at(cellIdOld+(nx-1)) == 2) {
            N={0.0,1.0,0.0};    
            S={0.0,UGD->y[j]+0.5*dy,0.0};
            s4 = -(Xold*N - S*N)/(U*N);
        }
        
        // -z normal (check cell at k-1)
        if(UGD->icellflag.at(cellIdOld-(nx-1)*(ny-1)) == 0 || UGD->icellflag.at(cellIdOld-(nx-1)*(ny-1)) == 2) {
            N={0.0,0.0,-1.0};    
            S={0.0,0.0,double(UGD->z_face[k-1])};
            s5 = -(Xold*N - S*N)/(U*N);
        }
        // +z normal (check cell at k+1)
        if(UGD->icellflag.at(cellIdOld+(nx-1)*(ny-1)) == 0 || UGD->icellflag.at(cellIdOld+(nx-1)*(ny-1)) == 2) {
            N={0.0,0.0,1.0};    
            S={0.0,0.0,double(UGD->z_face[k])};
            s6 = -(Xold*N - S*N)/(U*N);
        }
        
        // select which surface the particle is reflecting of:
        if((s1 < s) && (s1 >= -eps_S)) {
            s=s1;
            N={-1.0,0.0,0.0}; 
        }
        if((s2 < s) && (s2 >= -eps_S)) {
            s=s2;
            N={1.0,0.0,0.0}; 
        }
        if((s3 < s) && (s3 >= -eps_S)) {
            s=s3;
            N={0.0,-1.0,0.0}; 
        }
        if((s4 < s) && (s4 >= -eps_S)) {
            s=s4;
            N={0.0,1.0,0.0}; 
        }
        if((s5 < s) && (s5 >= -eps_S)) {
            s=s5;
            N={0.0,0.0,-1.0}; 
        }
        if((s6 < s) && (s6 >= -eps_S)) {
            s=s6;
            N={0.0,0.0,1.0}; 
        }
        
        
        if (s==100) {
            k = (int)(cellIdNew / ((nx-1)*(ny-1)));
            j = (int)((cellIdNew - k*(nx-1)*(ny-1))/(nx-1));
            i = cellIdNew -  j*(nx-1) - k*(nx-1)*(ny-1);
            
            //-x normal -> ckeck if i+1 is a building/terrain cell
            if(UGD->icellflag.at(cellIdNew+1) != 0 && UGD->icellflag.at(cellIdNew+1) != 2) {
                N={-1.0,0.0,0.0};    
                S={UGD->x[i]+0.5*dx,0.0,0.0};
                s1 = -(Xold*N - S*N)/(U*N);
            }
        
            //+x normal (i-1)
            if(UGD->icellflag.at(cellIdNew-1) != 0 && UGD->icellflag.at(cellIdNew-1) != 2) {
                N={1.0,0.0,0.0};    
                S={UGD->x[i]-0.5*dx,0.0,0.0};
                s2 = -(Xold*N - S*N)/(U*N);
            }
        
            //-y normal
            if(UGD->icellflag.at(cellIdNew+(nx-1)) != 0 && UGD->icellflag.at(cellIdNew+(nx-1)) != 2) {
                N={0.0,-1.0,0.0};
                S={0.0,UGD->y[j]+0.5*dy,0.0}; 
                s3 = -(Xold*N - S*N)/(U*N);
            }
        
            //+y normal
            if(UGD->icellflag.at(cellIdNew-(nx-1)) != 0 && UGD->icellflag.at(cellIdNew-(nx-1)) != 2) {
                N={0.0,1.0,0.0};    
                S={0.0,UGD->y[j]-0.5*dy,0.0};
                s4 = -(Xold*N - S*N)/(U*N);
            }
        
            //-z normal
            if(UGD->icellflag.at(cellIdNew+(nx-1)*(ny-1)) !=0 && UGD->icellflag.at(cellIdNew+(nx-1)*(ny-1)) != 2) {
                N={0.0,0.0,-1.0};    
                S={0.0,0.0,double(UGD->z_face[k])};
                s5 = -(Xold*N - S*N)/(U*N);
            }
            //+z normal
            if(UGD->icellflag.at(cellIdNew-(nx-1)*(ny-1)) !=0 && UGD->icellflag.at(cellIdNew-(nx-1)*(ny-1)) != 2) {
                N={0.0,0.0,1.0};
                S={0.0,0.0,double(UGD->z_face[k-1])};
                s6 = -(Xold*N - S*N)/(U*N);
            }
            
        
            // select which surface the particle is reflecting of:
            if((s1 < s) && (s1 >= -eps_S)){
                s=s1;
                N={-1.0,0.0,0.0}; 
            }
            if((s2 < s) && (s2 >= -eps_S)){
                s=s2;
                N={1.0,0.0,0.0}; 
            }
            if((s3 < s) && (s3 >= -eps_S)){
                s=s3;
                N={0.0,-1.0,0.0}; 
            }
            if((s4 < s) && (s4 >= -eps_S)){
                s=s4;
                N={0.0,1.0,0.0}; 
            }
            if((s5 < s) && (s5 >= -eps_S)){
                s=s5;
                N={0.0,0.0,-1.0}; 
            }
            if((s6 < s) && (s6 >= -eps_S)){
                s=s6;
                N={0.0,0.0,1.0}; 
            }
            
        
        // if no valid surface -> set particle to isActive=false
        if(l==100) {
            std::cout << i << " " << j << " " << k << std::endl;
            std::cout << s1 << " " << s2 << " " << s3 << " " << s4 << " " << s5 << " " << s6 << std::endl;
            std::cout << count << " " << cellFlagNew << " " << Xnew << Xold << U << std::endl;
            std::cout << "Reflection problem: no valid surface" << std::endl;
            return false;
        }
        
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
            std::cout << s1 << " " << s2 << " " << s3 << " " << s4 << " " << s5 << " " << s6 << std::endl;
            std::cout << cellFlagNew << Xnew << Xold << U << std::endl;
            std::cout << "Reflection problem: particle out of range" << std::endl;
            return false;
        }
        
    }

    if(UGD->icellflag.at(cellIdNew) == 0 || UGD->icellflag.at(cellIdNew) == 2) {
        std::cout << "Reflection problem: need more reflection" << std::endl;
        return false;
    }

    // if count run out -> set particle to isActive=false
    if(count>=25){
        std::cout << "Reflection problem: too many reflection" << std::endl;
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

