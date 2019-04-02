//
//  Advection.cpp
//  
//  This class handles advection of particles
//

#include "Advection.hpp"

Advection::Advection(Urb* urb, Turb* turb, Eulerian* eul, 
                     Dispersion* dis, PlumeInputData* PID) {
    
    std::cout<<"[Advection] \t Setting up sources "<<std::endl;
    
    
    Wind windRot;
    
    // make local copies
    nx = urb->grid.nx;
    ny = urb->grid.ny;
    nz = urb->grid.nz;
    
    numPar = PID->sources->numParticles;
    
    nBoxesX = PID->colParams->nBoxesX;
    nBoxesY = PID->colParams->nBoxesY;
    nBoxesZ = PID->colParams->nBoxesZ;
    
    boxSizeX = (PID->colParams->boxBoundsX2-PID->colParams->boxBoundsX1)/nBoxesX;	  
    boxSizeY = (PID->colParams->boxBoundsY2-PID->colParams->boxBoundsY1)/nBoxesY;	  
    boxSizeZ = (PID->colParams->boxBoundsZ2-PID->colParams->boxBoundsZ1)/nBoxesZ;
    
    volume=boxSizeX*boxSizeY*boxSizeZ;
    
    lBndx = PID->colParams->boxBoundsX1;
    uBndx = PID->colParams->boxBoundsX2;
    lBndy = PID->colParams->boxBoundsY1;
    uBndy = PID->colParams->boxBoundsY2;
    lBndz = PID->colParams->boxBoundsZ1;
    uBndz = PID->colParams->boxBoundsZ2;
    
    xBoxCen.resize(nBoxesX*nBoxesY*nBoxesZ);
    yBoxCen.resize(nBoxesX*nBoxesY*nBoxesZ);
    zBoxCen.resize(nBoxesX*nBoxesY*nBoxesZ);
    
    double quanX = (uBndx-lBndx)/(nBoxesX);
    double quanY = (uBndy-lBndy)/(nBoxesY);
    double quanZ = (uBndz-lBndz)/(nBoxesZ);
    
    int id=0;
    int zR=0;
    for(int k=0;k<nBoxesZ;++k) {
        int yR=0;
        for(int j=0;j<nBoxesY;++j) {
            int xR=0;
            for(int i=0;i<nBoxesX;++i) {
                id=k*nBoxesY*nBoxesX+j*nBoxesX+i;
                xBoxCen.at(id)=lBndx+xR*(quanX)+boxSizeX/2.0;
                yBoxCen.at(id)=lBndy+yR*(quanY)+boxSizeY/2.0;
                zBoxCen.at(id)=lBndz+zR*(quanZ)+boxSizeZ/2.0;	
                xR++;
            }
            yR++;
        }
        zR++;
    }
    
    tStepInp = PID->simParams->timeStep;
    avgTime  = PID->colParams->timeAvg;
    
    double sCBoxTime = PID->colParams->timeStart;
    int numTimeStep  = dis->numTimeStep;
    
    tStrt.resize(numPar);
    tStrt = dis->tStrt;
    
    timeStepStamp.resize(numTimeStep);
    timeStepStamp = dis->timeStepStamp;
    
    cBox.resize(nBoxesX*nBoxesY*nBoxesZ,0.0);
    
    int Flag=0;
    int flag=0;
    int flagPrime=0;
    int loopPrm=0;
    int loopLowestCell=0;
    int flag_g2nd=0;
    int countMax=10;//100;
    int countPrmMax=10;//1000;
    double ranU=0.0;
    double ranV=0.0;
    double ranW=0.0;
    int parPerTimestep=dis->parPerTimestep;
    int parToMove=0;
    
    // For every time step
    for(tStep=0; tStep<numTimeStep; tStep++) {
        
        // Move each particle for every time step
        parToMove = parToMove + parPerTimestep;
        
        for(int par=0; par<parToMove;par++) {
            loopPrm=0;
            
            int count=0;
            int countPrm=0;
            double xPos = dis->pos.at(par).x;
            double yPos = dis->pos.at(par).y;
            double zPos = dis->pos.at(par).z;
                        
            double tStepRem=tStepInp;
            double tStepUsed=0.0;
            double tStepCal=0.0;
            double dt=tStepRem;
            int loops=0;
            double tStepMin=tStepInp;
            int loopTby2=0;
            
            while(tStepRem>1.0e-5) { 
                int iV=int(xPos);
                int jV=int(yPos);
                int kV=int(zPos)+1;
                int id=kV*ny*nx+jV*nx+iV;
                
                // if the partice is in domain and ready to be released*/
                if(iV>0 && iV<nx-1 && jV>0 && jV<ny-1 && kV>0 && kV<nz-1 && urb->grid.icell.at(id)!=0) {
                        
	                loops++;                
                    
                    double eigVal_11=eul->eigVal.at(id).e11;
                    double eigVal_22=eul->eigVal.at(id).e22;
                    double eigVal_33=eul->eigVal.at(id).e33;
                    
                    double CoEps=turb->CoEps.at(id);
                    double tFac=0.5;
                    double tStepSigW=(2.0*(turb->sig.at(id).e33)*(turb->sig.at(id).e33)/CoEps);
                    double tStepEig11=-1.0/eigVal_11;
                    double tStepEig22=-1.0/eigVal_22;
                    double tStepEig33=-1.0/eigVal_33;
                    
                    double tStepArr[]={fabs(tStepEig11),fabs(tStepEig22),fabs(tStepEig33),fabs(tStepSigW)};
                    tStepCal=tFac * min(tStepArr,4); 
                    double arrT[]={tStepMin,tStepCal,tStepRem,dt};
                    dt=min(arrT,4);
                    double uPrime=dis->prime.at(par).x;
                    double vPrime=dis->prime.at(par).y;
                    double wPrime=dis->prime.at(par).z;
                    double uMean=urb->wind.at(id).u;
                    double vMean=urb->wind.at(id).v;
                    double wMean=urb->wind.at(id).w;
                    double ka0_11=eul->ka0.at(id).e11;
                    double ka0_21=eul->ka0.at(id).e21;
                    double ka0_31=eul->ka0.at(id).e31;
                    double g2nd_11=eul->g2nd.at(id).e11;
                    double g2nd_21=eul->g2nd.at(id).e21;
                    double g2nd_31=eul->g2nd.at(id).e31;
                    
                    double lam11=turb->lam.at(id).e11;
                    double lam12=turb->lam.at(id).e12;
                    double lam13=turb->lam.at(id).e13;
                    double lam21=turb->lam.at(id).e21;
                    double lam22=turb->lam.at(id).e22;
                    double lam23=turb->lam.at(id).e23;
                    double lam31=turb->lam.at(id).e31;
                    double lam32=turb->lam.at(id).e32;
                    double lam33=turb->lam.at(id).e33;
                    
                    double taudx11=eul->taudx.at(id).e11;
                    double taudx12=eul->taudx.at(id).e12;
                    double taudx13=eul->taudx.at(id).e13;
                    double taudx22=eul->taudx.at(id).e22;
                    double taudx23=eul->taudx.at(id).e23;
                    double taudx33=eul->taudx.at(id).e33;
                    
                    double taudy11=eul->taudy.at(id).e11;
                    double taudy12=eul->taudy.at(id).e12;
                    double taudy13=eul->taudy.at(id).e13;
                    double taudy22=eul->taudy.at(id).e22;
                    double taudy23=eul->taudy.at(id).e23;
                    double taudy33=eul->taudy.at(id).e33;
                    
                    double taudz11=eul->taudz.at(id).e11;
                    double taudz12=eul->taudz.at(id).e12;
                    double taudz13=eul->taudz.at(id).e13;
                    double taudz22=eul->taudz.at(id).e22;
                    double taudz23=eul->taudz.at(id).e23;
                    double taudz33=eul->taudz.at(id).e33;
                    
                    ranU=random::norRan();
                    double randXO=pow((CoEps*dt),0.5)*ranU;
                    double randXN=sqrt( (CoEps/(2.0*eigVal_11)) * ( exp(2.0*eigVal_11*dt)- 1.0 ) ) * ranU;
                    
                    ranV=random::norRan();
                    double randYO=pow((CoEps*dt),0.5)*ranV;
                    double randYN=sqrt( (CoEps/(2.0*eigVal_22)) * ( exp(2.0*eigVal_22*dt)- 1.0 ) ) * ranV;
                    
                    ranW=random::norRan();
                    double randZO=pow((CoEps*dt),0.5)*ranW;
                    double randZN=sqrt( (CoEps/(2.0*eigVal_33)) * ( exp(2.0*eigVal_33*dt)- 1.0 ) ) * ranW;                    
                    
                    eul->windP.e11=uPrime;
                    eul->windP.e21=vPrime;
                    eul->windP.e31=wPrime;
                    eul->windPRot=eul->matrixVecMult(eul->eigVecInv.at(id),eul->windP);
                      
                    double URot = eul->windPRot.e11;
                    double VRot = eul->windPRot.e21;
                    double WRot = eul->windPRot.e31;
                    
                    double URot_1st = URot*exp(eigVal_11*dt) - ( (ka0_11/eigVal_11)*( 1.0 - exp(eigVal_11*dt)) ) + randXN;
                    double VRot_1st = VRot*exp(eigVal_22*dt) - ( (ka0_21/eigVal_22)*( 1.0 - exp(eigVal_22*dt)) ) + randYN;
                    double WRot_1st = WRot*exp(eigVal_33*dt) - ( (ka0_31/eigVal_33)*( 1.0 - exp(eigVal_33*dt)) ) + randZN;
                      
                    eul->windPRot.e11 = URot_1st;
                    eul->windPRot.e21 = VRot_1st;
                    eul->windPRot.e31 = WRot_1st;
                    
                    eul->windP=eul->matrixVecMult(eul->eigVec.at(id),eul->windPRot);
                      
                    double U_1st = eul->windP.e11;
                    double V_1st = eul->windP.e21;
                    double W_1st = eul->windP.e31;
                    
                    flag_g2nd=0;
                    if(g2nd_11!=0.0 && U_1st!=0) {
                        if(g2nd_11/fabs(g2nd_11) == U_1st/fabs(U_1st)) {
                            flag_g2nd=1;
                        }
                    }
                    if(g2nd_21!=0.0 && V_1st!=0) {
                        if(g2nd_21/fabs(g2nd_21) == V_1st/fabs(V_1st)) {
                            flag_g2nd=1;
                        }
                    }
                    if(g2nd_31!=0.0 && W_1st!=0){
                        if(g2nd_31/fabs(g2nd_31) == W_1st/fabs(W_1st)) {
                            flag_g2nd=1;
                        }
                    }
                    if(flag_g2nd) { 
                        flag_g2nd=0;
                        double quan1=(1.0-g2nd_11*U_1st*dt);
                        double quan2=(1.0-g2nd_21*V_1st*dt);
                        double quan3=(1.0-g2nd_31*W_1st*dt);
                        
                        if(g2nd_11*U_1st!=0.0 && count<countMax) {
                            if(fabs(quan1)<0.5) {
                                tStepMin=2.0*dt;
                                count++;
                                continue;
                            }
                        }
                        if(g2nd_21*V_1st!=0.0 && count<countMax) {
                            if(fabs(quan2)<0.5) {
                                tStepMin=2.0*dt;
                                count++;
                                continue;
                            }
                        }
                        if(g2nd_31*W_1st!=0.0 && count<countMax) {
                            if(fabs(quan3)<0.5) {
                                tStepMin=dt*2.0;
                                count++;
                                continue;
                            }   
                        }   
                    }
                    double U_2nd = U_1st/(1.0-(g2nd_11*U_1st*dt));
                    double V_2nd = V_1st/(1.0-(g2nd_21*V_1st*dt));
                    double W_2nd = W_1st/(1.0-(g2nd_31*W_1st*dt));
                    
                    double du_3rd=0.5*( lam11*(taudy11*U_2nd*V_2nd + taudz11*U_2nd*W_2nd) 
                                       + lam12*(taudx11*V_2nd*U_2nd + taudy11*V_2nd*V_2nd + taudz11*V_2nd*W_2nd) 
                        	                + lam13*(taudx11*W_2nd*U_2nd + taudy11*W_2nd*V_2nd + taudz11*W_2nd*W_2nd) 
                        	                + lam21*(                      taudy12*U_2nd*V_2nd + taudz12*U_2nd*W_2nd)
                        	                + lam22*(taudx12*V_2nd*U_2nd + taudy12*V_2nd*V_2nd + taudz12*V_2nd*W_2nd) 
                        	                + lam23*(taudx12*W_2nd*U_2nd + taudy12*W_2nd*V_2nd + taudz12*W_2nd*W_2nd) 
                        	                + lam31*(                      taudy13*U_2nd*V_2nd + taudz13*U_2nd*W_2nd)
                        	                + lam32*(taudx13*V_2nd*U_2nd + taudy13*V_2nd*V_2nd + taudz13*V_2nd*W_2nd) 
                        	                + lam33*(taudx13*W_2nd*U_2nd + taudy13*W_2nd*V_2nd + taudz13*W_2nd*W_2nd) 
                        	               )*dt;
                    double dv_3rd=0.5*( lam11*(taudx12*U_2nd*U_2nd + taudy12*U_2nd*V_2nd + taudz12*U_2nd*W_2nd) 
                      	                + lam12*(taudx12*V_2nd*U_2nd +                       taudz12*V_2nd*W_2nd) 
                      	                + lam13*(taudx12*W_2nd*U_2nd + taudy12*W_2nd*V_2nd + taudz12*W_2nd*W_2nd) 
                      	                + lam21*(taudx22*U_2nd*U_2nd + taudy22*U_2nd*V_2nd + taudz22*U_2nd*W_2nd)
                      	                + lam22*(taudx22*V_2nd*U_2nd +                       taudz22*V_2nd*W_2nd) 
                      	                + lam23*(taudx22*W_2nd*U_2nd + taudy22*W_2nd*V_2nd + taudz22*W_2nd*W_2nd) 
                      	                + lam31*(taudx23*U_2nd*U_2nd + taudy23*U_2nd*V_2nd + taudz23*U_2nd*W_2nd)
                      	                + lam32*(taudx23*V_2nd*U_2nd +                       taudz23*V_2nd*W_2nd) 
                      	                + lam33*(taudx23*W_2nd*U_2nd + taudy23*W_2nd*V_2nd + taudz23*W_2nd*W_2nd) 
                                      )*dt;
                    double dw_3rd=0.5*( lam11*(taudx13*U_2nd*U_2nd + taudy13*U_2nd*V_2nd + taudz13*U_2nd*W_2nd) 
                      	                + lam12*(taudx13*V_2nd*U_2nd + taudy13*V_2nd*V_2nd + taudz13*V_2nd*W_2nd) 
                      	                + lam13*(taudx13*W_2nd*U_2nd + taudy13*W_2nd*V_2nd                      ) 
                      	                + lam21*(taudx23*U_2nd*U_2nd + taudy23*U_2nd*V_2nd + taudz23*U_2nd*W_2nd)
                      	                + lam22*(taudx23*V_2nd*U_2nd + taudy23*V_2nd*V_2nd + taudz23*V_2nd*W_2nd) 
                      	                + lam23*(taudx23*W_2nd*U_2nd + taudy23*W_2nd*V_2nd                      ) 
                      	                + lam31*(taudx33*U_2nd*U_2nd + taudy33*U_2nd*V_2nd + taudz33*U_2nd*W_2nd)
                      	                + lam32*(taudx33*V_2nd*U_2nd + taudy33*V_2nd*V_2nd + taudz33*V_2nd*W_2nd) 
                      	                + lam33*(taudx33*W_2nd*U_2nd + taudy33*W_2nd*V_2nd                      ) 
                                      )*dt;
                    uPrime=U_2nd+du_3rd;
                    vPrime=V_2nd+dv_3rd;
                    wPrime=W_2nd+dw_3rd;
                    
                    if(isnan(uPrime) || isnan(vPrime) || isnan(wPrime)) {
                        std::cerr<<"NAN.....>!!!!!!!"<<std::endl;
                        std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
                        std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
                        std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
                        std::cerr<<"tStep      : "<<tStep<<std::endl;
                        std::cerr<<"par        : "<<par<<std::endl;
                        std::cout<<"eigen Vector "<<std::endl;
                        eul->display(eul->eigVecInv.at(id));
                        std::cout<<"eigen Value  "<<std::endl;
                        eul->display(eul->eigVal.at(id));
                        exit(1);
                    }
                    
                    double terFacU = 2.5;
                    double terFacV = 2.;
                    double terFacW = 2.;
                    if(kV>10 && kV<14) {
                        terFacU = 8.5;
                        terFacV = 10.5;
                        terFacW = 10.5;
                    }
                    
                    if(fabs(uPrime)>terFacU*fabs(turb->sig.at(id).e11) && countPrm<countPrmMax) {
                        dis->prime.at(par).x = turb->sig.at(id).e11*random::norRan();
                        /*	      std::cout<<"Uprime OLD : "<<uPrime<<std::endl;
                        std::cout<<"Uprime New : "<<dis->prime.at(par).x<<std::endl;
                        std::cout<<"SIGMA U    : "<<turb->sig.at(id).e11<<std::endl;
                        std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
                        std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
                        std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
                        std::cout<<"Par        : "<<par<<std::endl;
                        std::cout<<"tStep      :  "<<tStep<<std::endl;*/
                        countPrm++;
                        flagPrime=1;
                    }
                    
                    if(fabs(vPrime)>terFacV*fabs(turb->sig.at(id).e22)&& countPrm<countPrmMax) {
                        dis->prime.at(par).y=turb->sig.at(id).e22*random::norRan();
                        /*	      std::cout<<"Vprime OLD : "<<vPrime<<std::endl;
                        std::cout<<"Vprime New : "<<dis->prime.at(par).y<<std::endl;
                        std::cout<<"SIGMA V    : "<<turb->sig.at(id).e22<<std::endl;
                        std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
                        std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
                        std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
                        std::cout<<"Par        : "<<par<<std::endl;
                        std::cout<<"tStep      :  "<<tStep<<std::endl;*/
                        countPrm++;
                        flagPrime=1;
                    }
                    
                    if(fabs(wPrime)>terFacW*fabs(turb->sig.at(id).e33)&& countPrm<countPrmMax){
                        dis->prime.at(par).z=turb->sig.at(id).e33*random::norRan();
                        /*	      std::cout<<"Wprime OLD : "<<wPrime<<std::endl;
                        std::cout<<"Wprime New : "<<dis->prime.at(par).z<<std::endl;
                        std::cout<<"SIGMA W    : "<<turb->sig.at(id).e33<<std::endl;
                        std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
                        std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
                        std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
                        std::cout<<"Par        : "<<par<<std::endl;
                        std::cout<<"tStep      :  "<<tStep<<std::endl;*/
                        countPrm++;
                        flagPrime=1;
                    }
                    
                    if(flagPrime==1) {
                        flagPrime=0;
                        loopPrm++;
                        continue;
                    }
                    
                    double disX=((uMean+uPrime)*dt);
                    double disY=((vMean+vPrime)*dt);
                    double disZ=((wMean+wPrime)*dt);
                    
                    if(fabs(disX)>eul->dx || fabs(disY)>eul->dy || fabs(disZ)>eul->dz){
                        tStepMin=dt/2.0;
                        loopTby2++;
                        continue;
                    }
                    
                    xPos=xPos+disX;
                    yPos=yPos+disY;
                    zPos=zPos+disZ;
                    
                    if(zPos<eul->zo) {
                        Flag=1;
                    }
                    
                    //reflection(zPos,wPrime,eul->zo,disX,disY,disZ,xPos,yPos,eul,iV,jV,kV,uPrime,vPrime);
                    
                    dis->prime.at(par).x=uPrime;
                    dis->prime.at(par).y=vPrime;
                    dis->prime.at(par).z=wPrime;
                    dis->pos.at(par).x=xPos;
                    dis->pos.at(par).y=yPos;
                    dis->pos.at(par).z=zPos;
                    
                    tStepUsed=tStepUsed+dt;
                    tStepRem=tStepRem-dt;
                    dt=tStepRem;
                    tStepMin=tStepInp;
                    loopTby2=0;
                } // if in domain
                else {
                    tStepRem=0.0;
                    dis->pos.at(par).x=-999.0;
                    dis->pos.at(par).y=-999.0;
                    dis->pos.at(par).z=-999.0;;
                }//if for domain ends   
            } // while (tStepRem>1.0e-5)
        } // for (int par=0; par<parToMove;par++)
        if(timeStepStamp.at(tStep) >= sCBoxTime){
            average(tStep,dis);
        }
        if(timeStepStamp.at(tStep)>= sCBoxTime+avgTime) {
            std::cout<<"loopPrm   :"<<loopPrm<<std::endl;
            std::cout<<"loopLowestCell :"<<loopLowestCell<<std::endl;
            //outputConc();
            avgTime=avgTime+PID->colParams->timeAvg;
        }
    } // for(tStep=0; tStep<numTimeStep; tStep++)
} // Advection()

double Advection::dot(const pos &vecA, const pos &vecB){
    return(vecA.x*vecB.x + vecA.y*vecB.y + vecA.z*vecB.z);
}

pos Advection::normalize(const pos &vec){
    double mag=sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z);
    pos vecTmp;
    vecTmp.x=vec.x/mag;
    vecTmp.y=vec.y/mag;
    vecTmp.z=vec.z/mag;
    return(vecTmp);
}

pos Advection::VecScalarMult(const pos &vec,const double &a){
    pos vecTmp;
    vecTmp.x=a*vec.x;
    vecTmp.y=a*vec.y;
    vecTmp.z=a*vec.z;
    return(vecTmp);
}

pos Advection::posAdd(const pos &vecA,const pos &vecB){
    pos vecTmp;
    vecTmp.x=vecA.x+vecB.x;
    vecTmp.y=vecA.y+vecB.y;
    vecTmp.z=vecA.z+vecB.z;
    return(vecTmp);
}

pos Advection::reflect(const pos &vec,const pos &normal) {
    pos a=VecScalarMult(normal , 2.0*dot(vec, normal));
    a.x=-a.x;
    a.y=-a.y;
    a.z=-a.z;
    pos vecTmp=posAdd(vec,a);
    return(vecTmp);
}

double Advection::distance(const pos &vecA,const pos &vecB) {
    return(sqrt((vecA.x-vecB.x)*(vecA.x-vecB.x) + (vecA.y-vecB.y)*(vecA.y-vecB.y) + (vecA.z-vecB.z)*(vecA.z-vecB.z) ));
}

pos Advection::posSubs(const pos &vecA, const pos & vecB){
    pos vecTmp;
    vecTmp.x=vecA.x-vecB.x;
    vecTmp.y=vecA.y-vecB.y;
    vecTmp.z=vecA.z-vecB.z;
    return(vecTmp);
}

void Advection::average(const int tStep,const Dispersion* dis) {
    for(int i=0;i<numPar;i++) {
        if(tStrt.at(i)>timeStepStamp.at(tStep)) continue;
        double xPos=dis->pos.at(i).x;
        double yPos=dis->pos.at(i).y;
        double zPos=dis->pos.at(i).z;
        if(zPos==-1) continue;
        int iV=int(xPos);
        int jV=int(yPos);
        int kV=int(zPos)+1;
        int idx=(int)((xPos-lBndx)/boxSizeX);
        int idy=(int)((yPos-lBndy)/boxSizeY);
        int idz=(int)((zPos-lBndz)/boxSizeZ);
        if(xPos<lBndx) idx=-1;
        if(yPos<lBndy) idy=-1;
        if(zPos<lBndz) idz=-1;
        int id=0;
        if(idx>=0 && idx<nBoxesX && idy>=0 && idy<nBoxesY && idz>=0 && idz<nBoxesZ && tStrt.at(i)<=timeStepStamp.at(tStep)) {
            id=idz*nBoxesY*nBoxesX+idy*nBoxesX+idx;
            cBox.at(id)=cBox.at(id)+1.0;
        }
    }
}

double Advection::min(double arr[],int len) {
    double min=arr[0];
    for(int i=1;i<len;i++) {
        if(arr[i]<min) {
            min=arr[i];
        }
    }
    return min;
}

double Advection::max(double arr[],int len) {
    double max=arr[0];
    for(int i=1;i<len;i++) {
        if(arr[i]>max) {
            max=arr[i];
        }
    }
    return max;
}
