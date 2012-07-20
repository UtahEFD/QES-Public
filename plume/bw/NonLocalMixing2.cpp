#include <iostream>
#include "NonLocalMixing2.h"


void nonLocalMixing2::createSigTau(eulerian* eul,util& utl){
    double vonKar=utl.vonKar;
    double cPope=.55;
    double sigUOrg=1.8;
    double sigVOrg=2.0;
    double sigWOrg=1.3;
    double sigUConst=1.5*sigUOrg*sigUOrg*cPope*cPope;//2.3438;
    double sigVConst=1.5*sigVOrg*sigVOrg*cPope*cPope;//1.5;
    double sigWConst=1.5*sigWOrg*sigWOrg*cPope*cPope;//0.6338;
    
    std::cout<<"IN Non-Local Mixing Func 2"<<std::endl;
    
    nx=utl.nx;
    ny=utl.ny;
    nz=utl.nz;
    
    dx=utl.dx;
    dy=utl.dy;
    dz=utl.dz;
    
    numBuild=utl.numBuild;
    xfo.resize(numBuild);
    yfo.resize(numBuild);
    zfo.resize(numBuild);
    hgt.resize(numBuild);
    wth.resize(numBuild);
    len.resize(numBuild);
    
    for(int i=0;i<numBuild;i++){
        xfo.at(i)=utl.xfo.at(i);
        yfo.at(i)=utl.yfo.at(i);
        zfo.at(i)=utl.zfo.at(i);
        hgt.at(i)=utl.hgt.at(i);
        wth.at(i)=utl.wth.at(i);
        len.at(i)=utl.len.at(i);
    }

    double constant=2.;
    double constant1=8.;
    for(int k=0;k<nz;k++){
        for(int j=0; j<ny;j++){
            for(int i=0;i<nx;i++){
                
                int id = k*ny*nx + j*nx + i;
                int idkp1=(k+1)*ny*nx + j*nx + i;
                int idkm1=(k-1)*ny*nx + j*nx + i;
                if(eul->CellType.at(id).c!=0){ //Calculate gradients ONLY if it is a fluid cell    
                    if(i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1){

                        //make sure that the diagonal taus are +ve and are greater than off diagonal
                        /*                            eul->tau.at(id).e11=eul->tau.at(id).e11+.5;
                            eul->tau.at(id).e22=eul->tau.at(id).e22+.5;
                            eul->tau.at(id).e33=eul->tau.at(id).e33+.5;
                            eul->tau.at(id).e12=eul->tau.at(id).e12+.5;
                            eul->tau.at(id).e23=eul->tau.at(id).e23+.5;
                            eul->tau.at(id).e13=eul->tau.at(id).e13+.5;
                        //make sure that the diagonal taus are greater than the off diagonal
                        
                            
                            eul->lam.at(id)=eul->matrixInv(eul->tau.at(id));
                            
                            
                            eul->sig.at(id).e11 = pow(eul->tau.at(id).e11,0.5);
                            eul->sig.at(id).e22 = pow(eul->tau.at(id).e22,0.5);
                            eul->sig.at(id).e33 = pow(eul->tau.at(id).e33,0.5);*/
                        //double Lm=eul->mixLen.at(id);
                        //double ustarCoEps=eul->sig.at(id).e33;
                        //eul->CoEps.at(id)=5.7* pow(ustarCoEps,3.0)/(Lm);*/
                        

                    }// if for domain
                }//if for celltype
            }
        }
    }
    
    
    /*     
    std::ofstream minDist;
    minDist.open("detMat.dat");
    if(!minDist.is_open()){
        std::cerr<<"CANNOT OPEN minDist"<<std::endl;
        exit(1);
    }
    
        
    std::ofstream taus;
    taus.open("taus.dat");
    
    std::ofstream points;
    points.open("points.dat");
    if(!points.is_open()){
        std::cerr<<"CANNOT OPEN points"<<std::endl;
        exit(1);
    }
    std::vector<double> Tv,Th;
    Tv.resize(nz*ny*nx);
    Th.resize(nz*ny*nx);
    
    for(int ibld=0;ibld<numBuild;ibld++){
        int iStrt = int(xfo.at(ibld));
        int iEnd  = int(xfo.at(ibld)+2.0*len.at(ibld));
        int jStrt = int(yfo.at(ibld)-(0.5*wth.at(ibld) + 3.0));
        int jEnd  = int(yfo.at(ibld)+(0.5*wth.at(ibld) + 3.0));
        int kEnd  = int(hgt.at(ibld)+0.5*hgt.at(ibld));
        
        for(int k = 0; k < kEnd;k++){
            for(int j = jStrt; j < jEnd;j++){
                for(int i = iStrt; i < iEnd;i++ ){
                    
                    int id = k*ny*nx+j*nx+i;

                    double tau11=0;
                    double tau22=0;
                    double tau33=0;
                    double tau12=0;
                    double tau23=0;
                    double tau13=0;
                    if(eul->CellType.at(id).c!=0){
                        
                        int idRefVert   = (kEnd+1)*nx*ny+j*nx+i;
                        int idRefHor    = k*nx*ny+(jEnd+1)*nx+i;
                        
                        double URefVert = eul->windVec.at(idRefVert).u;
                        double URefHor  = eul->windVec.at(idRefHor ).u;
                        
                        //double lRefVert = 0.75*hgt.at(ibld); //change at other occurance also, if changed here
                        //double lRefHor  = 0.5 *wth.at(ibld);

                        double lRefVert = fabs(k-kEnd);
                        double lRefHor  = fabs(j-jEnd);

                        double velGradVert= fabs(URefVert-eul->windVec.at(id).u);
                        double velGradHor = fabs(URefHor -eul->windVec.at(id).u);
                        
                        double velGrad = velGradVert;
                        double Lm = vonKar*lRefVert;
                        //double Lm = vonKar*getMinDistance(i,j,k);
                        
                        if(velGradHor>velGradVert){
                            velGrad=velGradHor;
                            Lm=vonKar*lRefHor;
                        }
                        
                        double S11 = velGrad/Lm;//dUdx;
                        double S22 = velGrad/Lm;//dVdy;
                        double S33 = velGrad/Lm;//dWdz;
                        double S12 = velGrad/Lm;//0.5*(dUdy+dVdx);
                        double S23 = velGrad/Lm;//0.5*(dVdz+dWdy);
                        double S13 = velGrad/Lm;//0.5*(dUdz+dWdx);
                        
                        double SijSij=S11*S11 + S22*S22 + S33*S33 + 2.0*(S12*S12 + S13*S13 + S23*S23);
                        
                        double nu_T = Lm*Lm * sqrt(2.0*SijSij);
                        
                        double Tke = pow( (nu_T/(cPope*Lm)) ,2.0);
                        double conNL=0.0005;
                        double conNL2=0.125;
                        double cond=0.4;
                        double condLow=0.7071;

                        tau11= conNL*( (2.0/3.0) * Tke - 2.0*(nu_T*S11) ) + eul->tau.at(id).e11;
                        if(eul->tau.at(id).e11>cond){
                            //                            tau11= conNL2*( (2.0/3.0) * Tke - 2.0*(nu_T*S11) ) - eul->tau.at(id).e11;
                            tau11= conNL2*eul->tau.at(id).e11;
                        }
                        if(eul->tau.at(id).e11<condLow){
                            tau11= condLow+eul->tau.at(id).e11;
                        }
                        tau22= conNL*( (2.0/3.0) * Tke - 2.0*(nu_T*S22) ) + eul->tau.at(id).e22;
                        if(eul->tau.at(id).e22>cond){
                            tau22= conNL2*eul->tau.at(id).e22;
                        }
                        if(eul->tau.at(id).e22<condLow){
                            tau22= condLow+eul->tau.at(id).e22;
                        }
                        tau33= conNL*( (2.0/3.0) * Tke - 2.0*(nu_T*S33) ) + eul->tau.at(id).e33;
                        if(eul->tau.at(id).e22>cond){
                            tau33= conNL2*eul->tau.at(id).e33;
                        }
                        if(eul->tau.at(id).e33<condLow){
                            tau33= condLow+eul->tau.at(id).e33;
                        }
                        
                        tau12= conNL*( - 2.0*(nu_T*S12)                 ) + eul->tau.at(id).e12;
                        if(eul->tau.at(id).e22>cond){
                            tau12= conNL2*eul->tau.at(id).e12;
                        }
                        if(eul->tau.at(id).e12<condLow){
                            tau12= condLow+eul->tau.at(id).e12;
                        }
                        tau13= conNL*( - 2.0*(nu_T*S13)                 ) + eul->tau.at(id).e13;
                        if(eul->tau.at(id).e22>cond){
                            tau13= conNL2*eul->tau.at(id).e13;
                        }
                        if(eul->tau.at(id).e13<condLow){
                            tau13= condLow+eul->tau.at(id).e13;
                        }
                        tau23= conNL*( - 2.0*(nu_T*S23)                 ) + eul->tau.at(id).e23;
                        if(eul->tau.at(id).e22>cond){
                            tau23= conNL2*eul->tau.at(id).e23;
                        }
                        if(eul->tau.at(id).e23<condLow){
                            tau23= condLow+eul->tau.at(id).e23;
                        }
                        
                        tau11=fabs(sigUConst*tau11);
                        tau22=fabs(sigVConst*tau22);
                        tau33=fabs(sigWConst*tau33);
                        
                        eul->tau.at(id).e11 = tau11; //adjust vertical gradients here
                        eul->tau.at(id).e22 = tau22;
                        eul->tau.at(id).e33 = tau33;
                        eul->tau.at(id).e12 = tau12;
                        eul->tau.at(id).e13 = tau13;
                        eul->tau.at(id).e23 = tau23;
                        eul->lam.at(id)=eul->matrixInv(eul->tau.at(id));
                        
                        eul->sig.at(id).e11 = pow(eul->tau.at(id).e11,0.5);
                        eul->sig.at(id).e22 = pow(eul->tau.at(id).e22,0.5);
                        eul->sig.at(id).e33 = pow(eul->tau.at(id).e33,0.5);
                        double ustarCoEps=sqrt(Tke)*cPope;
                        //            std::cout<<"ddddd"<<std::endl;
                        eul->CoEps.at(id)=5.7* pow(ustarCoEps,3.0)/(Lm);
                        
                        
                        //                        Tv.at(id) = fabs(URefVert-eul->windVec.at(id).u)/lRefVert;
                        //Th.at(id) = fabs(URefHor -eul->windVec.at(id).u)/lRefHor;
                        points<<i<<"  "<<j<<"  "<<k<<"  "<<std::endl;
                    }
                }
            }
            }*/
    }
    
    
    
    //points.close();
