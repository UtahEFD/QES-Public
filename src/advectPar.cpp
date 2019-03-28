#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>

#include "Eulerian.h"
#include "Dispersion.h"


typedef struct{
    double x;
    double y;
    double z;
}vec3;

void average(const int, const dispersion&);
void outputConc();
void reflection(double&, double&, const double&, const  double&,  const double&, const double&
		,double&,double&,const Eulerian&,const int&,const int&,const int&,double&,double&,const util&);
double dot(const vec3&, const vec3&);
vec3 normalize(const vec3&);
vec3 VecScalarMult(const vec3&,const double&);
vec3 reflect(const vec3&,const vec3&);
vec3 Vec3Subs(const vec3&,const vec3&);
vec3 Vec3Add(const vec3&,const vec3&);
double distance(const vec3&,const vec3&);

double min(double[],int);
double max(double[],int);



int numPar,nx,ny,nz,numBoxX,numBoxY,numBoxZ,tStep;
double xBoxSize,yBoxSize,zBoxSize,lBndx,lBndy,lBndz,uBndx,uBndy,uBndz,tStepInp,avgTime,volume;
std::vector<double> cBox,tStrt,timeStepStamp,xBoxCen,yBoxCen,zBoxCen;

std::ofstream output;
std::ofstream rand_output;
int loopExt=0;

void advectPar(const util &utl, dispersion &disp,Eulerian &eul, const char* model, const int argc){
  int parNo=-53;

  // Pete: Really need a singleton class to do this right... not like the commented out code below:
  // random::random();
  // 

  const char*  method=model;
  std::cout<<"Inside advect"<<std::endl;
  typedef struct{
    double u;
    double v;
    double w;
  }wind;
  wind windRot;

  numPar=utl.numPar; 
  
  nx=utl.nx;
  ny=utl.ny;
  nz=utl.nz;

  numBoxX=utl.numBoxX;
  numBoxY=utl.numBoxY;
  numBoxZ=utl.numBoxZ;

  xBoxSize = utl.xBoxSize;	  
  yBoxSize = utl.yBoxSize;	  
  zBoxSize = utl.zBoxSize;	  
  
  volume=xBoxSize*yBoxSize*zBoxSize;
  

  lBndx=utl.bnds[0];
  uBndx=utl.bnds[1];
  lBndy=utl.bnds[2];
  uBndy=utl.bnds[3];
  lBndz=utl.bnds[4];
  uBndz=utl.bnds[5];

  xBoxCen.resize(numBoxX*numBoxY*numBoxZ);
  yBoxCen.resize(numBoxX*numBoxY*numBoxZ);
  zBoxCen.resize(numBoxX*numBoxY*numBoxZ);

  
  double quanX=(uBndx-lBndx)/(numBoxX);
  double quanY=(uBndy-lBndy)/(numBoxY);
  double quanZ=(uBndz-lBndz)/(numBoxZ);

  int id=0;
  int zR=0;
  for(int k=0;k<numBoxZ;++k){
    int yR=0;
    for(int j=0;j<numBoxY;++j){
      int xR=0;
      for(int i=0;i<numBoxX;++i){
	id=k*numBoxY*numBoxX+j*numBoxX+i;
	
	xBoxCen.at(id)=lBndx+xR*(quanX)+xBoxSize/2.0;
	yBoxCen.at(id)=lBndy+yR*(quanY)+yBoxSize/2.0;

	zBoxCen.at(id)=lBndz+zR*(quanZ)+zBoxSize/2.0;	

	xR++;
      }
      yR++;
    }
    zR++;
  }
  
  tStepInp=utl.timeStep;
  avgTime=utl.avgTime;


  std::cout<<"NEXT:::::::::::::::"<<std::endl;
  if(argc==2 && std::strcmp(method,"o")==0){
    if(remove("output_old.m")!=0)
      perror("Output File Delete error");
    else
      std::cout<<" OLD - Output File succesfully removed!!"<<std::endl;
    output.open("output_old.m");
  }
  else{
    if(remove("output_new.m")!=0)
      perror("Output File Delete error");
    else
      std::cout<<" NEW - Output File succesfully removed!!"<<std::endl;
    
    output.open("output_new.m");
    rand_output.open("random.txt");
  }
  
  if(!output.is_open()){
    std::cerr<<"Output.dat file open error"<<std::endl;
    exit(1);
  }


  double sCBoxTime=utl.sCBoxTime;
  int numTimeStep=disp.numTimeStep;

  tStrt.resize(numPar);
  tStrt=disp.tStrt;

  timeStepStamp.resize(numTimeStep);
  timeStepStamp=disp.timeStepStamp;

  cBox.resize(numBoxX*numBoxY*numBoxZ,0.0);
  std::cout << "cBox.size = " << cBox.size() << "; " << numBoxX << " X " << numBoxY << " X " << numBoxZ << std::endl;

  std::ofstream particles;
  particles.open("particle.dat");
  if(!particles.is_open()){
      std::cerr<<"particle output file not open"<<std::endl;
      exit(1);
  }  
  std::ofstream outPrimes;
  
  outPrimes.open("Primes_new.dat");
  
  if(!outPrimes.is_open()){
    std::cerr<<"Prime output file not open"<<std::endl;
    exit(1);
  }  
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
  int parPerTimestep=disp.parPerTimestep;
  int parToMove=0;

  //For every time step
  for(tStep=0; tStep<numTimeStep; tStep++){
    std::cout<<"Time Step :                                 "<<tStep<<std::endl;

    // getchar();
    //Move each particle for every time step
    parToMove=parToMove+parPerTimestep;
    // std::cout<<parToMove<<std::endl;
    
    for(int par=0; par<parToMove;par++){
      loopPrm=0;
      // std::cout<<"par :                                     "<<par<<std::endl;
      
      int count=0;
      int countPrm=0;
      double xPos=disp.pos.at(par).x;
      double yPos=disp.pos.at(par).y;
      double zPos=disp.pos.at(par).z;
      if(par>9000)particles<<tStep<<"   "<<par<<"   "<<xPos<<"   "<<yPos<<"   "<<zPos<<std::endl;

      double tStepRem=tStepInp;
      double tStepUsed=0.0;
      double tStepCal=0.0;
      double dt=tStepRem;
      int loops=0;
      double tStepMin=tStepInp;
      int loopTby2=0;
      
      while(tStepRem>1.0e-5){

	int iV=int(xPos);
	int jV=int(yPos);
	int kV=int(zPos)+1;
	int id=kV*ny*nx+jV*nx+iV;
	
	
	/*if the partice is in domain and ready to be released*/
	//        if(par==20)std::cout<<tStep<<"   "<<par<<"   "<<xPos<<"   "<<yPos<<"   "<<zPos<<std::endl;
        //if(par==20)std::cout<<iV<<"   "<<jV<<"   "<<kV<<"   "<<nx<<"   "<<ny<<"   "<<nz<<"   "<<eul.CellType.at(id).c<<std::endl;
	//        std::cout<<tStep<<"   "<<par<<"   "<<xPos<<"   "<<yPos<<"   "<<zPos<<std::endl;
	if(iV>0 && iV<nx-1 && jV>0 && jV<ny-1 && kV>0 && kV<nz-1 && eul.CellType.at(id).c!=0){
	  
	    loops++;

	    
	    //	    std::cout<<par<<std::endl;
	    
	    double eigVal_11=eul.eigVal.at(id).e11;
	    double eigVal_22=eul.eigVal.at(id).e22;
	    double eigVal_33=eul.eigVal.at(id).e33;
	    
	    double CoEps=eul.CoEps.at(id);
	    double tFac=0.5;
	    double tStepSigW=(2.0*(eul.sig.at(id).e33)*(eul.sig.at(id).e33)/CoEps);
	    double tStepEig11=-1.0/eigVal_11;
	    double tStepEig22=-1.0/eigVal_22;
	    double tStepEig33=-1.0/eigVal_33;

	    double tStepArr[]={fabs(tStepEig11),fabs(tStepEig22),fabs(tStepEig33),fabs(tStepSigW)};
	    tStepCal=tFac * min(tStepArr,4); 
	    double arrT[]={tStepMin,tStepCal,tStepRem,dt};
	    dt=min(arrT,4);
	    double uPrime=disp.prime.at(par).x;
	    double vPrime=disp.prime.at(par).y;
	    double wPrime=disp.prime.at(par).z;
	    double uMean=eul.windVec.at(id).u;
	    double vMean=eul.windVec.at(id).v;
	    double wMean=eul.windVec.at(id).w;
	    double ka0_11=eul.ka0.at(id).e11;
	    double ka0_21=eul.ka0.at(id).e21;
	    double ka0_31=eul.ka0.at(id).e31;
	    double g2nd_11=eul.g2nd.at(id).e11;
	    double g2nd_21=eul.g2nd.at(id).e21;
	    double g2nd_31=eul.g2nd.at(id).e31;
	    
	    double lam11=eul.lam.at(id).e11;
	    double lam12=eul.lam.at(id).e12;
	    double lam13=eul.lam.at(id).e13;
	    double lam21=eul.lam.at(id).e21;
	    double lam22=eul.lam.at(id).e22;
	    double lam23=eul.lam.at(id).e23;
	    double lam31=eul.lam.at(id).e31;
	    double lam32=eul.lam.at(id).e32;
	    double lam33=eul.lam.at(id).e33;
	    
	    double taudx11=eul.taudx.at(id).e11;
	    double taudx12=eul.taudx.at(id).e12;
	    double taudx13=eul.taudx.at(id).e13;
	    double taudx22=eul.taudx.at(id).e22;
	    double taudx23=eul.taudx.at(id).e23;
	    double taudx33=eul.taudx.at(id).e33;

	    double taudy11=eul.taudy.at(id).e11;
	    double taudy12=eul.taudy.at(id).e12;
	    double taudy13=eul.taudy.at(id).e13;
	    double taudy22=eul.taudy.at(id).e22;
	    double taudy23=eul.taudy.at(id).e23;
	    double taudy33=eul.taudy.at(id).e33;
	    
	    double taudz11=eul.taudz.at(id).e11;
	    double taudz12=eul.taudz.at(id).e12;
	    double taudz13=eul.taudz.at(id).e13;
	    double taudz22=eul.taudz.at(id).e22;
	    double taudz23=eul.taudz.at(id).e23;
	    double taudz33=eul.taudz.at(id).e33;

	    
	    ranU=random::norRan();
	    double randXO=pow((CoEps*dt),0.5)*ranU;
	    double randXN=sqrt( (CoEps/(2.0*eigVal_11)) * ( exp(2.0*eigVal_11*dt)- 1.0 ) ) * ranU;
	    
	    ranV=random::norRan();
	    double randYO=pow((CoEps*dt),0.5)*ranV;
	    double randYN=sqrt( (CoEps/(2.0*eigVal_22)) * ( exp(2.0*eigVal_22*dt)- 1.0 ) ) * ranV;
	    
	    ranW=random::norRan();
	    double randZO=pow((CoEps*dt),0.5)*ranW;
	    double randZN=sqrt( (CoEps/(2.0*eigVal_33)) * ( exp(2.0*eigVal_33*dt)- 1.0 ) ) * ranW;

	    rand_output<<ranU<<" "<<ranV<<" "<<ranW<<"\n ";
	    
	    if(argc==2 && std::strcmp(method,"o")==0){//old method
	      if(tStep==0 && par==0)
		std::cout<<"OLD METHOD"<<std::endl;
	      double du=(-0.5*CoEps*(lam11*uPrime+lam13*wPrime) )*dt + 0.5*taudz13*dt +
		taudz11*lam11*uPrime*wPrime*0.5*dt + taudz11*lam13*wPrime*wPrime*0.5*dt + 
		taudz13*lam13*uPrime*wPrime*0.5*dt + taudz13*lam33*wPrime*wPrime*0.5*dt +  randXO;
	      
	      double dv=(-0.5*CoEps*(lam22*vPrime) )*dt + taudz22*lam22*vPrime*0.5*wPrime*dt +  randYO;
	      
	      double dw=(-0.5*CoEps*(lam13*uPrime+lam33*wPrime) )*dt +(0.5*taudz33*dt)+
		taudz13*lam11*uPrime*0.5*wPrime*dt + taudz13*lam13*wPrime*0.5*wPrime*dt + 
		taudz33*lam13*uPrime*0.5*wPrime*dt + taudz33*lam33*wPrime*0.5*wPrime*dt +  randZO;
	      
	      uPrime=(uPrime+du);
	      vPrime=(vPrime+dv);
	      wPrime=(wPrime+dw);
	      
	      if(isnan(uPrime) || isnan(vPrime) || isnan(wPrime)){
		std::cerr<<"NAN.....>!!!!!!!"<<std::endl;
		std::cerr<<"tStep:  "<<tStep<<std::endl;
		std::cerr<<"par   :"<<par<<std::endl;
		exit(1);
	      }
	      
	    }// if condition ends for old method
	    if(argc!=2 || std::strcmp(method,"new")==0){//new method
	      
	      if(tStep==0 && par==0)
		std::cout<<"NEW METHOD"<<std::endl;
	      
	      eul.windP.e11=uPrime;
	      eul.windP.e21=vPrime;
	      eul.windP.e31=wPrime;
	      
	      if(par==parNo)
		std::cout<<"Initial:  "<<eul.windP.e11<<"    "<<eul.windP.e21<<"    "<<eul.windP.e31<<std::endl;
	      
	      eul.windPRot=eul.matrixVecMult(eul.eigVecInv.at(id),eul.windP);
	      
	      double URot=eul.windPRot.e11;
	      double VRot=eul.windPRot.e21;
	      double WRot=eul.windPRot.e31;
	      
	      
	      double URot_1st=URot*exp(eigVal_11*dt) - ( (ka0_11/eigVal_11)*( 1.0 - exp(eigVal_11*dt)) ) + randXN;
	      double VRot_1st=VRot*exp(eigVal_22*dt) - ( (ka0_21/eigVal_22)*( 1.0 - exp(eigVal_22*dt)) ) + randYN;
	      double WRot_1st=WRot*exp(eigVal_33*dt) - ( (ka0_31/eigVal_33)*( 1.0 - exp(eigVal_33*dt)) ) + randZN;
	      
	      
	      eul.windPRot.e11=URot_1st;
	      eul.windPRot.e21=VRot_1st;
	      eul.windPRot.e31=WRot_1st;
	      
	      eul.windP=eul.matrixVecMult(eul.eigVec.at(id),eul.windPRot);
	      
	      double U_1st=eul.windP.e11;
	      double V_1st=eul.windP.e21;
	      double W_1st=eul.windP.e31;
	      
	      flag_g2nd=0;
	      if(g2nd_11!=0.0 && U_1st!=0){
		if(g2nd_11/fabs(g2nd_11) == U_1st/fabs(U_1st))
		  flag_g2nd=1;
	      }
	      if(g2nd_21!=0.0 && V_1st!=0){
		if(g2nd_21/fabs(g2nd_21) == V_1st/fabs(V_1st))
		  flag_g2nd=1;
	      }
	      if(g2nd_31!=0.0 && W_1st!=0){
		if(g2nd_31/fabs(g2nd_31) == W_1st/fabs(W_1st))
		  flag_g2nd=1;
	      }
	      
	      
	      if(flag_g2nd){
		flag_g2nd=0;
		
		double quan1=(1.0-g2nd_11*U_1st*dt);
		double quan2=(1.0-g2nd_21*V_1st*dt);
		double quan3=(1.0-g2nd_31*W_1st*dt);
		
		if(count>0 && count<200) {
		  //std::cout<<".";
		}
		else if(int(count)%100==0 ){
		  //std::cout<<".";
		}
		
		if(g2nd_11*U_1st!=0.0 && count<countMax){
		  if(fabs(quan1)<0.5){
		    tStepMin=2.0*dt;
		    count++;
		    continue;
		  }
		}
		if(g2nd_21*V_1st!=0.0 && count<countMax){
		  if(fabs(quan2)<0.5){
		    tStepMin=2.0*dt;
		    count++;
		    continue;
		  }
		}
		if(g2nd_31*W_1st!=0.0 && count<countMax){
		  if(fabs(quan3)<0.5){
		    tStepMin=dt*2.0;
		    count++;
		    continue;
		  }
		}
		
	      }
	      else{
		//std::cout<<"Passed the test"<<g2nd_31<<"   "<<W_1st<<std::endl;
	      }// if condition ends for second step falg
	      
	      double U_2nd=U_1st/(1.0-(g2nd_11*U_1st*dt));
	      double V_2nd=V_1st/(1.0-(g2nd_21*V_1st*dt));
	      double W_2nd=W_1st/(1.0-(g2nd_31*W_1st*dt));
	    
	      
	      
	      double du_3rd=0.5*(  lam11*(                      taudy11*U_2nd*V_2nd + taudz11*U_2nd*W_2nd) 
				   + lam12*(taudx11*V_2nd*U_2nd + taudy11*V_2nd*V_2nd + taudz11*V_2nd*W_2nd) 
				   + lam13*(taudx11*W_2nd*U_2nd + taudy11*W_2nd*V_2nd + taudz11*W_2nd*W_2nd) 
				   + lam21*(                      taudy12*U_2nd*V_2nd + taudz12*U_2nd*W_2nd)
				   + lam22*(taudx12*V_2nd*U_2nd + taudy12*V_2nd*V_2nd + taudz12*V_2nd*W_2nd) 
				   + lam23*(taudx12*W_2nd*U_2nd + taudy12*W_2nd*V_2nd + taudz12*W_2nd*W_2nd) 
				   + lam31*(                      taudy13*U_2nd*V_2nd + taudz13*U_2nd*W_2nd)
				   + lam32*(taudx13*V_2nd*U_2nd + taudy13*V_2nd*V_2nd + taudz13*V_2nd*W_2nd) 
				   + lam33*(taudx13*W_2nd*U_2nd + taudy13*W_2nd*V_2nd + taudz13*W_2nd*W_2nd) 
				   )*dt;
	      double dv_3rd=0.5*(  lam11*(taudx12*U_2nd*U_2nd + taudy12*U_2nd*V_2nd + taudz12*U_2nd*W_2nd) 
				   + lam12*(taudx12*V_2nd*U_2nd +                       taudz12*V_2nd*W_2nd) 
				   + lam13*(taudx12*W_2nd*U_2nd + taudy12*W_2nd*V_2nd + taudz12*W_2nd*W_2nd) 
				   + lam21*(taudx22*U_2nd*U_2nd + taudy22*U_2nd*V_2nd + taudz22*U_2nd*W_2nd)
				   + lam22*(taudx22*V_2nd*U_2nd +                       taudz22*V_2nd*W_2nd) 
				   + lam23*(taudx22*W_2nd*U_2nd + taudy22*W_2nd*V_2nd + taudz22*W_2nd*W_2nd) 
				   + lam31*(taudx23*U_2nd*U_2nd + taudy23*U_2nd*V_2nd + taudz23*U_2nd*W_2nd)
				   + lam32*(taudx23*V_2nd*U_2nd +                       taudz23*V_2nd*W_2nd) 
				   + lam33*(taudx23*W_2nd*U_2nd + taudy23*W_2nd*V_2nd + taudz23*W_2nd*W_2nd) 
				   )*dt;
	      double dw_3rd=0.5*(  lam11*(taudx13*U_2nd*U_2nd + taudy13*U_2nd*V_2nd + taudz13*U_2nd*W_2nd) 
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
	      
	      if(par==parNo){
		std::cout<<"Par :"<<par<<std::endl;
		std::cout<<"tStep :"<<tStep<<"          "<<dt<<std::endl;
		std::cout<<"xPos :"<<xPos<<"     "<<iV<<std::endl;
		std::cout<<"yPos :"<<yPos<<"     "<<jV<<std::endl;
		std::cout<<"zPos :"<<zPos<<"     "<<kV<<std::endl;
		std::cout<<"Umean :"<<uMean<<std::endl;
		std::cout<<"Vmean :"<<vMean<<std::endl;
		std::cout<<"Wmean :"<<wMean<<std::endl;
		
		std::cout<<"First Rotation:  "<<URot<<"  "<< VRot<<"   "<< WRot<<std::endl;

		std::cout<<"After Rotation:  "<<URot_1st<<"  "<< VRot_1st<<"   "<< WRot_1st<<std::endl;
		std::cout<<"Rotation to original :  "<<U_1st<<"  "<< V_1st<<"   "<< W_1st<<std::endl;
		std::cout<<"Second Step:  "<<U_2nd<<"  "<< V_2nd<<"   "<< W_2nd<<std::endl;
		std::cout<<"Last increment :  "<<du_3rd<<"  "<< dv_3rd<<"   "<< dw_3rd<<std::endl;
		std::cout<<"Final:  "<<uPrime<<"   "<<vPrime<<"   "<<wPrime<<std::endl;
		
		//		std::cout<<"Taudx :"<<std::endl;
		//eul.display(eul.taudx.at(id));
		//std::cout<<std::endl;
		//
		//std::cout<<"Taudy :"<<std::endl;
		//eul.display(eul.taudy.at(id));
		//std::cout<<std::endl;
		//
		//std::cout<<"Taudz :"<<std::endl;
		//eul.display(eul.taudz.at(id));
		//std::cout<<std::endl;
		//
		//std::cout<<"Tau :"<<std::endl;
		//eul.display(eul.tau.at(id));
		//std::cout<<std::endl;
		//
		//std::cout<<"lam :"<<std::endl;
		//eul.display(eul.lam.at(id));
		//std::cout<<std::endl;
		//getchar();
	      }
	      
	      if(isnan(uPrime) || isnan(vPrime) || isnan(wPrime)){
		std::cerr<<"NAN.....>!!!!!!!"<<std::endl;
		std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
		std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
		std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
		
		std::cerr<<"tStep      : "<<tStep<<std::endl;
		std::cerr<<"par        : "<<par<<std::endl;
		std::cout<<"eigen Vector "<<std::endl;
		//eul.display(eul.eigVecInv.at(id));
		std::cout<<"eigen Value  "<<std::endl;
		//eul.display(eul.eigVal.at(id));
		exit(1);
	      }
	      
	    } // if condition ends for new method
	    
	    double terFacU = 2.5;
            double terFacV = 2.;
            double terFacW = 2.;
            if(kV>10 && kV<14){
                //terFacU = 8.5;
                //terFacV = 10.5;
                //terFacW = 10.5;
                
            }
            
	    if(fabs(uPrime)>terFacU*fabs(eul.sig.at(id).e11) && countPrm<countPrmMax){
	      
	      disp.prime.at(par).x=eul.sig.at(id).e11*random::norRan();
	      /*	      std::cout<<"Uprime OLD : "<<uPrime<<std::endl;
	      std::cout<<"Uprime New : "<<disp.prime.at(par).x<<std::endl;
	      std::cout<<"SIGMA U    : "<<eul.sig.at(id).e11<<std::endl;
	      std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
	      std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
	      std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
	      std::cout<<"Par        : "<<par<<std::endl;
	      std::cout<<"tStep      :  "<<tStep<<std::endl;*/
	      countPrm++;
	      flagPrime=1;
	    }
	    if(fabs(vPrime)>terFacV*fabs(eul.sig.at(id).e22)&& countPrm<countPrmMax){
	      disp.prime.at(par).y=eul.sig.at(id).e22*random::norRan();
	      /*	      std::cout<<"Vprime OLD : "<<vPrime<<std::endl;
	      std::cout<<"Vprime New : "<<disp.prime.at(par).y<<std::endl;
	      std::cout<<"SIGMA V    : "<<eul.sig.at(id).e22<<std::endl;
	      std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
	      std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
	      std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
	      std::cout<<"Par        : "<<par<<std::endl;
	      std::cout<<"tStep      :  "<<tStep<<std::endl;*/
	      countPrm++;
	      flagPrime=1;
	    }
	    if(fabs(wPrime)>terFacW*fabs(eul.sig.at(id).e33)&& countPrm<countPrmMax){
	      disp.prime.at(par).z=eul.sig.at(id).e33*random::norRan();
	      
	      /*	      std::cout<<"Wprime OLD : "<<wPrime<<std::endl;
	      std::cout<<"Wprime New : "<<disp.prime.at(par).z<<std::endl;
	      std::cout<<"SIGMA W    : "<<eul.sig.at(id).e33<<std::endl;
	      std::cout<<"xPos       : "<<xPos<< "    "<<iV<<std::endl;
	      std::cout<<"yPos       : "<<yPos<< "    "<<jV<<std::endl;
	      std::cout<<"zPos       : "<<zPos<< "    "<<kV<<std::endl;
	      std::cout<<"Par        : "<<par<<std::endl;
	      std::cout<<"tStep      :  "<<tStep<<std::endl;*/
	      countPrm++;
	      flagPrime=1;
	    }
	    if(flagPrime==1){
	      flagPrime=0;
	      loopPrm++;
	      //	    getchar();
	      continue;
	    }
	    double disX=((uMean+uPrime)*dt);
	    double disY=((vMean+vPrime)*dt);
	    double disZ=((wMean+wPrime)*dt);
	    
	    
	    if(fabs(disX)>utl.dx || fabs(disY)>utl.dy || fabs(disZ)>utl.dz){
	      tStepMin=dt/2.0;
	      loopTby2++;
	      continue;
	    }
            /*	    if(count>0 && loopPrm>0 && tStep==53){
	      std::cout<<"\n"<<"count: "<<count<<"  "<<par<<"  "<<tStep<<" "<<dt<<"  "<<loopPrm<< "  "<< iV<<"  "<<jV<<"  "<<kV<<std::endl;
	      std::cout<<"eigen Value  "<<std::endl;
	      eul.display(eul.eigVal.at(id));
              }*/
	    xPos=xPos+disX;
	    yPos=yPos+disY;
	    zPos=zPos+disZ;
	    
	    
	    if(zPos<utl.zo){
	      Flag=1;
	    }


	    reflection(zPos,wPrime,utl.zo,disX,disY,disZ,xPos,yPos,eul,iV,jV,kV,uPrime,vPrime,utl);
	    
	    if(Flag==1){
	      if(par==parNo){
		std::cout<<"After Reflection :"<<zPos<<"   "<<wPrime<<std::endl;
		//getchar();
	      }
	      Flag=0;
	      if(zPos>1.0){
		std::cout<<"zPos is reflected off the ground!! GREATER THAN 1 grid cell"<<std::endl;
		//getchar();
	      }
	    }
	    
	    disp.prime.at(par).x=uPrime;
	    disp.prime.at(par).y=vPrime;
	    disp.prime.at(par).z=wPrime;
	    
	    disp.pos.at(par).x=xPos;
	    disp.pos.at(par).y=yPos;
	    disp.pos.at(par).z=zPos;
	    
	    tStepUsed=tStepUsed+dt;
	    tStepRem=tStepRem-dt;
	    dt=tStepRem;
	    tStepMin=tStepInp;
	    loopTby2=0;
	}// if condition for domain---SEE ELSE ALSO 
	else{
	  tStepRem=0.0;
	  disp.pos.at(par).x=-999.0;
	  disp.pos.at(par).y=-999.0;
	  disp.pos.at(par).z=-999.0;;
	  //	  std::cout<<xPos<<"   "<<yPos<<"   "<<zPos<<"   "<<par<<"   "<<tStep<<std::endl;
	  //getchar();
	}//if for domain ends
	
	
      }//while for time
      //if(1)outPrimes<<tStep<<"  "<<par<<"  "<<xPos<<"  "<<yPos<<"  "<<zPos<<std::endl;
      
    }//particle
    
    if(timeStepStamp.at(tStep) >= sCBoxTime){
      average(tStep,disp);
    }
    if(timeStepStamp.at(tStep)>= sCBoxTime+avgTime){
      std::cout<<"loopPrm   :"<<loopPrm<<std::endl;
      std::cout<<"loopLowestCell :"<<loopLowestCell<<std::endl;
      outputConc();
      avgTime=avgTime+utl.avgTime;
    }
  }//time step
}//advect par

void reflection(double &zPos, double &wPrime, const double &z0,const double &disX
		,const double &disY,const double &disZ ,double &xPos
		,double &yPos,const Eulerian &eul, const int &imc, const int &jmc
		, const int &kmc,double &uPrime,double &vPrime,const util &utl ){
    
    //shader reflection   
    //Now do Reflection		
    //	vec3 u;
    //point of intersection
    //	vec3 pI;	
    //incident vector
    //	vec3 l;
    //reflection vector
    //	vec3 r;
    //normal vector
    //	vec3 normal;
    //distance from reflected surface
    //	float dis;
    
    //	float denom;
    //	float numer;
    
    //	vec2 cIndex;

    
    vec3 u,n,vecS,prevPos,normal,vecZ,vecTmp,pI,r,l,pos,prmCurr;
    double d,denom,numer,dis;
    pos.x=xPos;
    pos.y=yPos;
    pos.z=zPos;
            
    prevPos.x=xPos-disX;
    prevPos.y=yPos-disY;
    prevPos.z=zPos-disZ;

    prmCurr.x=uPrime;
    prmCurr.y=vPrime;
    prmCurr.z=wPrime;
    
    int i = int(xPos);
    int j = int(yPos);
    int k = int(zPos)+1;
    int id=0;
    
    double eps_S = 0.0001;
    double eps_d = 0.01;
    double smallestS = 100.0;
    double cellBld = 1.0;
    
    if((i < nx) && (j < ny) && (k < nz) && (i >= 0) && (j >= 0)){ //check if within domain
        double cellBld = 1.0; //set it so that default is no reflection
        
        if(k < 0)//ground is at 0 in this code, this also avoids out of bound in case of large negative k
            k = 0;

	id=k*nx*ny+j*nx+i;

        if(k >= 0){
            //Perform lookup into wind texture to see if new position is inside a building
            cellBld = eul.CellBuild.at(id).c;
        }
	int count=0;
        while((eul.CellType.at(id).c==0 || (zPos < 0.0)) && count<25){ //pos.z<0.0 covers ground reflections

	  /*	  std::cout<<"Before Reflection-prev"<<xPos-disX<<"   "<<yPos-disY<<"   "<<zPos-disZ<<std::endl;
	  std::cout<<std::endl;
	  std::cout<<"Before Reflection"<<xPos<<"   "<<yPos<<"   "<<zPos<<std::endl;
	  std::cout<<"Before Reflection"<<uPrime<<"   "<<vPrime<<"   "<<wPrime<<std::endl;*/


	  //	  std::cout<<"reflection while, count:"<<count<<std::endl;
	  count=count+1;
	  
	  u.x =disX;// vec3(pos) - prevPos;// u has disX,disY and disZ
	  u.y =disY;
	  u.z =disZ;
          
	  double s1 = -1.0; //for -x
	  double s2 = -1.0; //for +x
	  double s3 = -1.0; //for -y
	  double s4 = -1.0; //for +y
	  double s5 = -1.0; //for +z (buildings)
	  double s6 = -1.0; //Not used
	  double s7 = -1.0; //for ground
          
	  smallestS = 100.0;
	  double xfo=-999.0;
	  double yfo=-999.0;
	  double zfo=-999.0;
	  double ht =-999.0;
	  double wti=-999.0;
	  double lti=-999.0;

	  //	  std::cout<<"reflection while, build params,cellBld:"<<cellBld<<"  "<<id<<"  "<<i<<"  "<<j<<"  "<<k<<std::endl;
          if(cellBld!=-1){
	    xfo = utl.xfo.at(int(cellBld));//bcoords.x;
	    yfo = utl.yfo.at(int(cellBld));//bcoords.y;
	    zfo = utl.zfo.at(int(cellBld));//bcoords.z;
	    ht  = utl.hgt.at(int(cellBld));//bdim.x;
	    wti = utl.wth.at(int(cellBld));//bdim.y;
	    lti = utl.len.at(int(cellBld));//bdim.z;
	  }
	  //	  std::cout<<"xfo:"<<"  "<<xfo<<" yfo:"<<"  "<<yfo<<" zfo:"<<"  "<<zfo<<std::endl;
          
	  //-x normal  
	  n.x=-1.0;
	  n.y=0.0;
	  n.z=0.0;
	  vecS.x=xfo;
	  vecS.y=0.0;
	  vecS.z=0.0;
	  
	  d = -dot(n,vecS);
	  denom = dot(n,u);
	  numer = dot(n,prevPos) + d;
	  s1 = -numer/denom;

	  /*	  std::cout<<"-x:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
	  std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
	  std::cout<<"-x:s1:"<<"  "<<s1<<std::endl;*/
          
	  //+x normal
	  n.x=1.0;
	  n.y=0.0;
	  n.z=0.0;
	  vecS.x=xfo+lti;
	  vecS.y=0.0;
	  vecS.z=0.0;
	  d = -dot(n,vecS);
	  denom = dot(n,u);
	  numer = dot(n,prevPos) + d;
	  s2 = -numer/denom;

	  /*std::cout<<"+x:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
	  std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
	  std::cout<<"s2:"<<"  "<<s2<<std::endl;*/
	  
	  //+y normal
	  n.x=0.0;
	  n.y=1.0;
	  n.z=0.0;
	  
	  vecS.x=xfo;
	  vecS.y=yfo+(wti/2.0);
	  vecS.z=0.0;
          
	  d = -dot(n,vecS);
	  denom = dot(n,u);
	  numer = dot(n,prevPos) + d;
	  s3 = -numer/denom;

	  /*std::cout<<"+y:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
	  std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
	  std::cout<<"s3:"<<"  "<<s3<<std::endl;*/
          
	  //-y normal
	  n.x=0.0;
	  n.y=-1.0;
	  n.z=0.0;
	  
	  vecS.x=xfo;
	  vecS.y=yfo-(wti/2.0);
	  vecS.z=0.0;
	  d = -dot(n,vecS);
	  denom = dot(n,u);
	  numer = dot(n,prevPos) + d;
	  s4 = -numer/denom;
	  
	  /*std::cout<<"-y:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
	  std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
	  std::cout<<"s4:"<<"  "<<s4<<std::endl;*/

	  //+z normal
	  n.x=0.0;
	  n.y=0.0;
	  n.z=1.0;
	  
	  vecS.x=xfo;
	  vecS.y=0.0;
	  vecS.z=zfo+ht;
          
	  d = -dot(n,vecS);
	  denom = dot(n,u);
	  numer = dot(n,prevPos) + d;
	  s5 = -numer/denom;

	  //std::cout<<"+z:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
	  //std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
	  //std::cout<<"s5:"<<"  "<<s5<<std::endl;
	  
	  //Ground plane
	  n.x=0.0;
	  n.y=0.0;
	  n.z=1.0;
	  numer = dot(n,prevPos);
	  denom = dot(n,u);
	  s7 = -numer/denom;

	  //std::cout<<"gr:n:"<<"  "<<n.x<<"   "<<n.y<<"  "<<n.z<<std::endl;
	  //std::cout<<"vecS:"<<"  "<<vecS.x<<"   "<<vecS.y<<"  "<<vecS.z<<std::endl;
	  //std::cout<<"s7:"<<"  "<<s7<<std::endl;
          
          
	  if((s1 < smallestS) && (s1 >= -eps_S)){
	    smallestS = s1;
	    normal.x=-1.0;
	    normal.y=0.0;
	    normal.z=0.0;
	    //std::cout<<"s1-smallestS:"<<"  "<<smallestS<<std::endl;
	  }
	  if((s2 < smallestS) && (s2 >= -eps_S)){
	    normal.x =1.0;
	    normal.y=0.0;
	    normal.z=0.0;
	    smallestS = s2;
	    //std::cout<<"s2-smallestS:"<<"  "<<smallestS<<std::endl;
	  }
	  if((s3 < smallestS) && (s3 >= -eps_S)){
	    normal.x=0.0;
	    normal.y=1.0;
	    normal.z=0.0;
	    smallestS = s3;
	    //std::cout<<"s3-smallestS:"<<"  "<<smallestS<<std::endl;
	  }	
	  if((s4 < smallestS) && (s4 >= -eps_S)){
	    normal.x=0.0;
	    normal.y=-1.0;
	    normal.z=0.0;
	    smallestS = s4;
	    //std::cout<<"s4-smallestS:"<<"  "<<smallestS<<std::endl;
	  }	   
	  if((s5 < smallestS) && (s5 >= -eps_S)){
	    normal.x =0.0;
	    normal.y=0.0;
	    normal.z=1.0;
	    smallestS = s5;
	    //std::cout<<"s5-smallestS:"<<"  "<<smallestS<<std::endl;
	  }	 
	  //std::cout<<"normal:"<<"  "<<normal.x<<"   "<<normal.y<<"  "<<normal.z<<std::endl;
	  //std::cout<<"smallestS:"<<"  "<<smallestS<<std::endl;
	 



	  //Detect Edge Collision

	  double edgeS = fabs(smallestS-s7);
	  //std::cout<<"edgeS:"<<"  "<<edgeS<<", eps_d:"<<eps_d<<std::endl;
	  if((edgeS < eps_d)){
	    //smallestS = s6;
	    vecZ.x=0.0;
	    vecZ.y=0.0;
	    vecZ.z=1.0;
	    vecTmp.x=normal.x+vecZ.x;
	    vecTmp.y=normal.y+vecZ.y;
	    vecTmp.z=normal.z+vecZ.z;
	    //std::cout<<"first cond"<<std::endl;
	    
	    normal = normalize(vecTmp);
	  }
	  else if((s7 < smallestS) && (s7 >= -eps_S)){
	    //std::cout<<"else cond"<<std::endl;
	    normal.x=0.0;
	    normal.y=0.0;
	    normal.z=1.0;
	    smallestS = s7;
	  }
	  //std::cout<<"edge and else:normal:"<<"  "<<normal.x<<"   "<<normal.y<<"  "<<normal.z<<std::endl;
	  //std::cout<<"after edge: smallestS:"<<"  "<<smallestS<<std::endl;

	  vec3 vecTmp1=VecScalarMult(u,smallestS);
	  pI=Vec3Add(vecTmp1,prevPos);
	  //std::cout<<"pI:"<<"  "<<pI.x<<"   "<<pI.y<<"  "<<pI.z<<std::endl;
          
	  if((smallestS >= -eps_S) && (smallestS <= eps_S)){
	    pI = prevPos;
	    r = normal;
	  }	
	  else{
	    l = normalize(Vec3Subs(pI,prevPos));
	    r = normalize(reflect(l,normal));
	  }
	  //std::cout<<"l:"<<"  "<<l.x<<"   "<<l.y<<"  "<<l.z<<std::endl;
	  //std::cout<<"r:"<<"  "<<r.x<<"   "<<r.y<<"  "<<r.z<<std::endl;
	  
	  dis = distance(pI,pos);		
	  //std::cout<<"dis:"<<"  "<<dis<<std::endl;
	  
	  prevPos = pI;
	  pos = Vec3Add(pI,VecScalarMult(r,dis));
	  //update xpos,ypos,zpos
	  xPos=pos.x;
	  yPos=pos.y;
	  zPos=pos.z;
	  prmCurr = reflect(prmCurr,normal);
	  uPrime=prmCurr.x;
	  vPrime=prmCurr.y;
	  wPrime=prmCurr.z;
	  //update primes
	  i = int(pos.x);
	  j = int(pos.y);
	  k = int(pos.z)+1;

          
	  //NOTE: Consider what happens if building is too close to domain.
	  //Do check to make sure i,j,k's are valid;
	  cellBld = 1.0;
	  if(k < 0)
	    k = 0;

	  id=k*nx*ny+j*nx+i;
	  if(k >= 0){
	    //std::cout<<"end of while cellbld check"<<i<<"  "<<j<<"   "<<k<<std::endl;
	    cellBld = eul.CellBuild.at(id).c;  //find cellType of (i,j,k) {cellType stores
	  }
	  //std::cout<<"after Reflection"<<xPos<<"   "<<yPos<<"   "<<zPos<<std::endl;
	  //std::cout<<"after Reflection"<<uPrime<<"   "<<vPrime<<"   "<<wPrime<<std::endl;
          //          std::cout<<"count:"<<count<<std::endl;
	  if(smallestS>=99.999 || count>20){
	     std::cout<<"may be a reflection problem"<<std::endl;
	    std::cout<<"count:"<<count<<std::endl;
	    std::cout<<"smallestS:"<<smallestS<<std::endl;
	    //getchar();
	  }
	  


	  //getchar();
        }//while loop for reflection
    }//domain check
    //shader reflfection ends
    
    
    //    std::cout<<"out reflection"<<std::endl;
    
    return;
}

double dot(const vec3 &vecA, const vec3 &vecB){
    return(vecA.x*vecB.x + vecA.y*vecB.y + vecA.z*vecB.z);
}
vec3 normalize(const vec3 &vec){
    double mag=sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z);
    vec3 vecTmp;
    
    vecTmp.x=vec.x/mag;
    vecTmp.y=vec.y/mag;
    vecTmp.z=vec.z/mag;
    return(vecTmp);
}
vec3 VecScalarMult(const vec3 &vec,const double &a){
    vec3 vecTmp;
    vecTmp.x=a*vec.x;
    vecTmp.y=a*vec.y;
    vecTmp.z=a*vec.z;
    return(vecTmp);
}

vec3 Vec3Add(const vec3 &vecA,const vec3 &vecB){
    vec3 vecTmp;
    vecTmp.x=vecA.x+vecB.x;
    vecTmp.y=vecA.y+vecB.y;
    vecTmp.z=vecA.z+vecB.z;
    return(vecTmp);
}

vec3 reflect(const vec3 &vec,const vec3 &normal){
    vec3 a=VecScalarMult(normal , 2.0*dot(vec, normal));
    a.x=-a.x;
    a.y=-a.y;
    a.z=-a.z;
    vec3 vecTmp=Vec3Add(vec,a);
    return(vecTmp);
}

double distance(const vec3 &vecA,const vec3 &vecB){
    return(sqrt((vecA.x-vecB.x)*(vecA.x-vecB.x) + (vecA.y-vecB.y)*(vecA.y-vecB.y) + (vecA.z-vecB.z)*(vecA.z-vecB.z) ));
}
vec3 Vec3Subs(const vec3 &vecA,const vec3 & vecB){
    vec3 vecTmp;
    vecTmp.x=vecA.x-vecB.x;
    vecTmp.y=vecA.y-vecB.y;
    vecTmp.z=vecA.z-vecB.z;
    return(vecTmp);
}


void average(const int tStep,const dispersion &disp){
  
  for(int i=0;i<numPar;i++){
    if(tStrt.at(i)>timeStepStamp.at(tStep))
	continue;
    double xPos=disp.pos.at(i).x;
    double yPos=disp.pos.at(i).y;
    double zPos=disp.pos.at(i).z;
    if(zPos==-1)
      continue;
    
    int iV=int(xPos);
    int jV=int(yPos);
    int kV=int(zPos)+1;


    
    int idx=(int)((xPos-lBndx)/xBoxSize);
    int idy=(int)((yPos-lBndy)/yBoxSize);
    int idz=(int)((zPos-lBndz)/zBoxSize);
    
    if(xPos<lBndx)
      idx=-1;
    if(yPos<lBndy)
      idy=-1;
    if(zPos<lBndz)
      idz=-1;
    
    
    
    int id=0;
    if(idx>=0 && idx<numBoxX && idy>=0 && idy<numBoxY && idz>=0 && idz<numBoxZ && tStrt.at(i)<=timeStepStamp.at(tStep)){
      id=idz*numBoxY*numBoxX+idy*numBoxX+idx;
      cBox.at(id)=cBox.at(id)+1.0;
    }
    
  }
}
void outputConc(){
  std::cout<<"output"<<std::endl;
  std::cout<<"Time:  "<<tStep<<std::endl;
  output << "data = [" << std::endl;
  double conc=(tStepInp)/(avgTime*volume*numPar);
  for(int k=0;k<numBoxZ;k++)
    for(int j=0;j<numBoxY;j++)
      for(int i=0;i<numBoxX;i++){
	int id=k*numBoxY*numBoxX+j*numBoxX+i;
      	output<<xBoxCen.at(id)<<" "<<yBoxCen.at(id)<<" "<<zBoxCen.at(id)<<" "<<cBox.at(id)*conc<< ';' << std::endl;
  }
  output << "];" << std::endl;
  for(int k=0;k<numBoxZ;k++)
    for(int j=0;j<numBoxY;j++)
      for(int i=0;i<numBoxX;i++){
	int id=k*numBoxY*numBoxX+j*numBoxX+i;
	
	cBox.at(id)=0.0;
      }

  // the following is a mechanism to plot the data nicely in matlab
  output << "[aa bb] = size(data); " << std::endl;
  output << "x = unique(data(:,1)); " << std::endl;
  output << "y = unique(data(:,2)); " << std::endl;
  output << "z = unique(data(:,3));" << std::endl;
  output << "nx = length(x);" << std::endl;
  output << "ny = length(y);" << std::endl;
  output << "for zht = 1:length(z)    %% or, you can select the z-height at which you want concentration contours " << std::endl;
  output << "   cc=1;" << std::endl;
  output << "   conc_vector_zht=0;" << std::endl;
  output << "   for ii = 1:aa " << std::endl;
  output << "      if data(ii,3) == z(zht,:)" << std::endl;
  output << "         conc_vector_zht(cc,1) = data(ii,4);" << std::endl;
  output << "         cc=cc+1;" << std::endl;
  output << "      end" << std::endl;
  output << "   end" << std::endl;
  output << "   conc_matrix_zht=0; " << std::endl;
  output << "   conc_matrix_zht = reshape(conc_vector_zht,nx,ny)';" << std::endl;
  output << "   figure(zht)" << std::endl;
  output << "   h = pcolor(x,y,log10(conc_matrix_zht));" << std::endl;
  output << "   set(h,'edgecolor','none');" << std::endl;
  output << "   shading interp;" << std::endl;
  output << "   hh=colorbar;" << std::endl;
  output << "   set(get(hh,'ylabel'),'string','log10(Concentration)','fontsize',20);" << std::endl;
  output << "   set(gcf,'color','w');" << std::endl;
  output << "   set(gcf,'visible','off'); %%this is to make sure the image is not displayed" << std::endl;
  output << "   xlabel('$x$','interpreter','latex','fontsize',20,'color','k'); " << std::endl;
  output << "   ylabel('$y$','interpreter','latex','fontsize',20,'color','k');" << std::endl;
  output << "   caxis([-8 3.5]);" << std::endl;
  output << "   string = strcat('log10(Concentration) Contours; Horizontal x-y plane; Elevation z = ',num2str(z(zht,:)));" << std::endl;
  output << "   h=title(string,'fontsize',12);" << std::endl;
  output << "   axis equal;" << std::endl;

  // % hold on 
  // % for ii = 1:Bldsize %%%% plot as many buildings as there are in the domain
  // %     xvalues = [Xfo(ii) Xfo(ii)+Length(ii) Xfo(ii)+Length(ii) Xfo(ii)];
  // %     yvalues = [Yfo(ii)-(Width(ii) / 2) Yfo(ii)-(Width(ii) / 2) Yfo(ii)+(Width(ii) / 2) Yfo(ii)+(Width(ii) / 2)];
  // %     fill(xvalues,yvalues,[0.8 0.8 0.6]);
  // %     hold on
  // % end

  output << "   filename = sprintf('concentrationData_zht=%05.1f.png', z(zht,:));" << std::endl;
  output << "   print('-dpng', filename);" << std::endl;
  output << "end" << std::endl;
}


double min(double arr[],int len){
  
  double min=arr[0];
  for(int i=1;i<len;i++){
    if(arr[i]<min)
      min=arr[i];
  }
  
  return min;
}

double max(double arr[],int len){
  
  double max=arr[0];
  for(int i=1;i<len;i++){
    if(arr[i]>max)
      max=arr[i];
  }
  
  return max;
}
