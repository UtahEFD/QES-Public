#include "Plume.hpp"

void Plume::reflection(URBGeneralData* UGD, Eulerian* eul,
                       double& xPos, double& yPos, double& zPos, 
                       double& disX, double& disY, double& disZ,
                       double& uFluct, double& vFluct, double& wFluct)
{
    
    const double eps_S = 0.0001;
    
    int cellId=eul->getCellId(xPos,yPos,zPos); 
    int currentCellFlag=UGD->icellflag.at(cellId);
    
    Vector3<double> v1,v2;

    Vector3<double> vecFluct;
    Vector3<double> X,P,S;
    Vector3<double> U,V1,V2;
    Vector3<double> R,N;

    // vector of distance travelled by particle 
    U[0]=disX;
    U[1]=disY;
    U[2]=disZ;
    //std::cout << "U = " << U << std::endl;
    // vector postion of the particle before step 
    X[0]=xPos-disX;
    X[1]=yPos-disY;
    X[2]=zPos-disZ;
    //std::cout << "X = " << X << std::endl;
    // vector of fluctuation
    vecFluct[0]=uFluct;
    vecFluct[1]=vFluct;
    vecFluct[2]=wFluct;

    //std::cout << "particle position: (" << xPos << "," << yPos << "," << zPos << ")" << std::endl;
    
    int count=0;
    while( (currentCellFlag==0 || currentCellFlag==2 ) && count<25){ //pos.z<0.0 covers ground reflections
    
        double d=0.0,s=0.0;

        double s1 = -1.0; //for -x
        double s2 = -1.0; //for +x
        double s3 = -1.0; //for -y
        double s4 = -1.0; //for +y
        double s5 = -1.0; //for -z
        double s6 = -1.0; //for +z

        //-x normal
        //N={-1.0,0.0,0.0};    
        //S={0.0,0.0,0.0}  
        //s1 = -(X*N + S*N)/(U*N);
        
        //+x normal
        //N={-1.0,0.0,0.0};    
        //S={0.0,0.0,0.0}
        //s2 = -(X*N + S*N)/(U*N);

        //-y normal
        //N={0.0,-1.0,0.0};    
        //S={0.0,0.0,0.0}  
        //s1 = -(X*N + S*N)/(U*N);
        
        //+y normal
        //N={0.0,1.0,0.0};    
        //S={0.0,0.0,0.0}  
        //s1 = -(X*N + S*N)/(U*N);
        
        
        //-z normal
        N={0.0,0.0,-1.0};    
        S={0.0,0.0,0.0};
        s5 = -(X*N + S*N)/(U*N);

        // select which surface the particle is reflecting of:
        //FM -> for noe only flat ground is implemented
        s=s5;
        N={0.0,0.0,-1.0};    
        
        // vector from current postition to the wall
        V1=s*U;
        // postion of relfection on the wall
        P=X+V1;
        // distance traveled after the wall
        V2=U-V1;
        d=V2.length();
        // reflection: normalizing V2 -> R is of norm 1
        V2=V2/V2.length();
        R=reflect(V2,N);
        // update postion from surface reflection
        X=P+d*R;

        // relfection of the Fluctuation 
        vecFluct = reflect(vecFluct,N);

        // increment the relfection count
        count=count+1;                
        // update the icellflag
        try {
            cellId=eul->getCellId(X[0],X[1],X[2]);
            currentCellFlag=UGD->icellflag.at(cellId);
        } catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << std::endl;
            std::cerr << "particle position after reflection: "<< X << std::endl;
            exit(1);
        }
        
    }
    if(count>=25){
        std::cout<<"may be a reflection problem"<<std::endl;
        std::cout<<"count:"<<count<<std::endl;
    }
    
    // update output variable
    // particle position
    xPos=X[0];
    yPos=X[1];
    zPos=X[2];
    // fluctuation
    uFluct=vecFluct[0];
    vFluct=vecFluct[1];
    wFluct=vecFluct[2];    
    
    return;
}

