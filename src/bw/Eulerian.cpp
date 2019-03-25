#include <iostream>
#include <ctime>
#include <cmath>

#include "Eulerian.h"
#include "Random.h"
#include "Turbulence.h"
#include "LocalMixing.h"
#include "NonLocalMixing.h"
#include "NonLocalMixing2.h"


eulerian::eulerian() : windField(0),zo(0.0),dz(0.0){}

void eulerian::createEul(const util& u){
    std::cout<<"in createeul "<<std::clock()<<std::endl;
  utl=u;
  zo=utl.zo;
  dz=utl.dz;
  windField=utl.windFieldData;
  nx=utl.nx;
  ny=utl.ny;
  nz=utl.nz;
  vonKar=utl.vonKar;

  zInMeters.resize(nz,0.0);
  
  for(int k=0;k<nz;k++)
  {
    zInMeters.at(k)= -0.5*dz + k*dz;//k==0 means ground
    //-0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 ... 25.5 26.5 27.5 28.5
    std::cout<<zInMeters.at(k)<<" ";
  }

  createWindField();
  
}

void eulerian::createWindField(){//currently just call windFromQUIC;
    std::cout<<"in createwindfield "<<std::clock()<<std::endl;
  switch(windField){
  case 3:
    uniform();
    break;
  case 4:
    shear();
    break;
  case 5:
    windFromQUIC();
    break;
  case 6:
    readCellType();
    windFromQUIC();
    break;
  case 10:
    uniform();
    break;
  default:
    std::cerr<<"NO WIND Field Specified"<<std::endl;
    exit(1);
  }

}

void eulerian::uniform(){
        std::cout<<"in uniform"<<std::endl;
  
  windVec.resize(nx*ny*nz);
  CellType.resize(nx*ny*nz);
  
  for(int k=0;k<nz; k++)
    for(int j=0;j<ny;j++)
      for(int i=0;i<nx;i++){
	int id=k*ny*nx + j*nx + i;
	if(k==0){//ground
	  windVec.at(id).u=0.0;
	  windVec.at(id).v=0.0;
	  windVec.at(id).w=0.0;
	  
	  CellType.at(id).c = 0;
	}
	else{
	  windVec.at(id).u=2.0;
	  windVec.at(id).v=0.0;
	  windVec.at(id).w=0.0;
	  
	  CellType.at(id).c = 1;
	}
      }
  createSigmaAndEps();
}
void eulerian::shear(){
        std::cout<<"in shear"<<std::endl;
  windVec.resize(nx*ny*nz);
  double Wh= 3.0;//Change in sigma function too if you change values here
  double Zh=10.0;
  for(int k=0;k<nz; k++)
    for(int j=0;j<ny;j++)
      for(int i=0;i<nx;i++){
	int id=k*ny*nx + j*nx + i;
	if(k==0){//ground
	  windVec.at(id).u=0.0;
	  windVec.at(id).v=0.0;
	  windVec.at(id).w=0.0;
	}
	else{
	  windVec.at(id).u=Wh*( log((zInMeters.at(k))/zo) / log(Zh/zo) );
	  windVec.at(id).v=0.0;
	  windVec.at(id).w=0.0;
	}
      }
  createSigmaAndEps();
}

void eulerian::windFromQUIC(){//read wind field data from file

  windVec.resize(nx*ny*nz);
  
  std::ifstream QUICWindField;
  std::string velocityField_filename = utl.m_QUICProjData.m_quicProjectPath + "QU_velocity.dat";
  std::cout << "Attempting to open velocity field: " << velocityField_filename << std::endl;
  QUICWindField.open(velocityField_filename.c_str());

  if(!QUICWindField.is_open()){
    std::cerr<<"Wind input File open error"<<std::endl;
    exit(1);
  }
  std::string header;
  
  QUICWindField>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header
  	       >>header>>header>>header>>header>>header>>header>>header>>header>>header>>header;


  double quicIndex;
  int id;
  for(int k = 0; k < nz; k++)
    for(int j = 0; j < ny; j++)
      for(int i = 0; i < nx; i++){
	 id= k*nx*ny + j*nx + i;

	QUICWindField>>quicIndex; // ignoring the X,Y and Z values
	QUICWindField>>quicIndex;
	QUICWindField>>quicIndex;
	
	QUICWindField>>windVec.at(id).u ;//storing the velocity values in the wind structure
	QUICWindField>>windVec.at(id).v ;
	QUICWindField>>windVec.at(id).w ;
      }
  
  std::cout<<windVec.at(id).u<<" "<<windVec.at(id).v<<" "<<windVec.at(id).w<<" "<<"\n";
  std::cout<<"num of windfielddata is:"<<id<<"\n";
  QUICWindField.close();
  createSigmaAndEps();
}

void eulerian::readCellType(){
    std::cout<<"read cell type "<<std::clock()<<std::endl;
  CellType.resize(nx*ny*nz);
  
  std::ifstream QUICCellType;
  QUICCellType.open("../plume/bw/QU_celltype.dat");

  if(!QUICCellType.is_open()){
    std::cerr<<"CellType input File open error"<<std::endl;
    exit(1);
  }
  std::string header;
  
  double quicIndex;
  int id;
  for(int k = 0; k < nz; k++) //k==0 is ground
    for(int j = 0; j < ny; j++)
      for(int i = 0; i < nx; i++){
	id = k*nx*ny + j*nx + i;

	QUICCellType>>quicIndex; // ignoring the X,Y and Z values
	QUICCellType>>quicIndex;
	QUICCellType>>quicIndex;
	QUICCellType>>CellType.at(id).c ;//storing the velocity values in the wind structure
// 	std::cout<<CellType.at(id).c <<"  "<<id<<"\n";
      } 
  std::cout<<"num of QUICCellType is:"<<id<<"\n";
  QUICCellType.close();
  addBuildingsInWindField();
}


void eulerian::addBuildingsInWindField(){
    std::cout<<"create cellBuild satrts "<<std::endl;
    
    CellBuild.resize(nx*ny*nz);
    
    for(int k = 0; k < nz; k++){   //k==0 is ground
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                int id = k*nx*ny + j*nx + i;
                CellBuild.at(id).c =- 1;
            }
        }
    }
    std::cout<<"create cellBuild :1 "<<std::endl;
    
    int numBuild=utl.numBuild;
    
    std::cout<<"create cellBuild, building: "<<std::endl;
    for(int n=0; n < numBuild; n++){
      //        std::cout<<"create cellBuild, building: "<<n<<std::endl;
      int lk = int(utl.zfo.at(n));
      int uk = int(utl.zfo.at(n)+utl.hgt.at(n))+1;//added 1 as the loop below has < sign and building go to 10
      int lj = int(utl.yfo.at(n)-(utl.wth.at(n)/2.0));
      int uj = int(utl.yfo.at(n)+(utl.wth.at(n)/2.0));
      int li = int(utl.xfo.at(n));
      int ui = int(utl.xfo.at(n)+utl.len.at(n));
      //        std::cout<<"create cellBuild,after build data "<<std::endl;
      for(int k= lk; k < uk; k++){
	for(int j= lj; j < uj; j++){
	  for(int i= li; i < ui; i++){
	    //std::cout<<"create cellBuild:i,j,k "<<i<<","<<j<<","<<k<<std::endl;
	    int id = k*nx*ny + j*nx + i;
	    if(CellType.at(id).c==0)
	    {
	      CellBuild.at(id).c = n;
// 	      std::cout<<"create cellBuild:i,j,k "<<i<<","<<j<<","<<k<<std::endl;
	    }
	  }
	}
      }
    }
    /*    std::ofstream cellBld;
	  cellBld.open("cellBuild.dat");
	  for(int k = 0; k < nz; k++){   
	  for(int j = 0; j < ny; j++){
	  for(int i = 0; i < nx; i++){
	  int id = k*nx*ny + j*nx + i;
	  cellBld<<i<<"   "<<j<<"   "<<k<<"   "<<CellBuild.at(id).c<<std::endl;
	  }
	  }
	  }
	  cellBld.close();*/
    std::cout<<"create cellBuild end "<<std::clock()<<std::endl;
    return;
}

 
void eulerian::createSigmaAndEps(){
        std::cout<<"in createsigmaand eps"<<std::endl;
  
  CoEps.resize(nx*ny*nz);
  sig.resize(nx*ny*nz);
  tau.resize(nx*ny*nz);
  lam.resize(nx*ny*nz);
  std::cout<<"INside sig and eps"<<std::endl;
  
   turbulence* turb;

  
  
  switch(windField){
  case 6:
      //      createSigmaAndEpsQUICFull();
      turb = new localMixing;//non localMixing turbulence
      turb->createSigTau(this,utl);
      delete turb;
      
      //      turb = new nonLocalMixing;
      //turb->createSigTau(this,utl);
      //delete turb;

      turb = new nonLocalMixing2;
      turb->createSigTau(this,utl);
      delete turb;

      
    
    break;

  case 3: //uniform
    for(int k=0;k<nz;++k)
      for(int j=0;j<ny;++j)
	for(int i=0; i<nx;i++){
	  int id=k*ny*nx + j*nx + i;
	  
	  
	  sig.at(id).e11 = 2.0 * utl.ustar;
	  sig.at(id).e12 = 0.0;
	  sig.at(id).e13 = 0.0;//utl.ustar*utl.ustar;
	  sig.at(id).e22 = 1.6 * utl.ustar;
	  sig.at(id).e23 = 0.0;
	  sig.at(id).e33 = 1.2 * utl.ustar;
	  
	  CoEps.at(id)=5.7*pow(utl.ustar,3.0) / (vonKar*nz/2.0);
	  
	  tau.at(id).e11=sig.at(id).e11*sig.at(id).e11;
	  tau.at(id).e12=sig.at(id).e12*sig.at(id).e12;
	  tau.at(id).e13=sig.at(id).e13*sig.at(id).e13;

	  tau.at(id).e22=sig.at(id).e22*sig.at(id).e22;
	  tau.at(id).e23=sig.at(id).e23*sig.at(id).e23;
	  tau.at(id).e33=sig.at(id).e33*sig.at(id).e33;
	 	  
	  double detTau=(tau.at(id).e11*tau.at(id).e22*tau.at(id).e33)-
	    (tau.at(id).e11*tau.at(id).e23*tau.at(id).e23)-
	    (tau.at(id).e12*tau.at(id).e12*tau.at(id).e33)+
	    (tau.at(id).e12*tau.at(id).e23*tau.at(id).e13)+
	    (tau.at(id).e13*tau.at(id).e12*tau.at(id).e23)-
	    (tau.at(id).e13*tau.at(id).e22*tau.at(id).e13);
	 	  
	  lam.at(id).e11=( (tau.at(id).e22*tau.at(id).e33)-(tau.at(id).e23*tau.at(id).e23) )/detTau;
	  lam.at(id).e12=( (tau.at(id).e13*tau.at(id).e23)-(tau.at(id).e12*tau.at(id).e33) )/detTau;
	  lam.at(id).e13=( (tau.at(id).e12*tau.at(id).e23)-(tau.at(id).e13*tau.at(id).e22) )/detTau;
	  lam.at(id).e21=( (tau.at(id).e23*tau.at(id).e13)-(tau.at(id).e12*tau.at(id).e33) )/detTau;
	  lam.at(id).e22=( (tau.at(id).e11*tau.at(id).e33)-(tau.at(id).e13*tau.at(id).e13) )/detTau;
	  lam.at(id).e23=( (tau.at(id).e13*tau.at(id).e12)-(tau.at(id).e11*tau.at(id).e23) )/detTau;
	  lam.at(id).e31=( (tau.at(id).e12*tau.at(id).e23)-(tau.at(id).e22*tau.at(id).e13) )/detTau;
	  lam.at(id).e32=( (tau.at(id).e12*tau.at(id).e13)-(tau.at(id).e11*tau.at(id).e23) )/detTau;
	  lam.at(id).e33=( (tau.at(id).e11*tau.at(id).e22)-(tau.at(id).e12*tau.at(id).e12) )/detTau;	  
	}
    break;
  case 5:
  case 4://shear
    createUstar();
    for(int k=0;k<nz;++k)
      for(int j=0;j<ny;++j)
	for(int i=0; i<nx;++i){

	  int id=k*ny*nx + j*nx + i;

	  //double ustarEul=vonKar*windVec.at(id).w/ (log(zInMeters.at(k)/zo));
	  double ustarEul=ustar.at(id);
	  double vertShear=pow( ( 1.0-( zInMeters.at(k)/(nz) ) ), 3.0/4.0);
	  
	  sig.at(id).e11 = 2.5 * ustarEul;// * vertShear;
	  sig.at(id).e12 = 0.0;
	  sig.at(id).e13 = ustarEul;//*vertShear;
	  sig.at(id).e22 = 2.3 * ustarEul;// * vertShear;
	  sig.at(id).e23 = 0.0;
	  sig.at(id).e33 = 1.3 * ustarEul;// * vertShear;


	  
	  CoEps.at(id)=5.7*pow(ustarEul,3.0) / (vonKar*(zInMeters.at(k)+zo));

	  tau.at(id).e11=sig.at(id).e11*sig.at(id).e11;
	  tau.at(id).e12=sig.at(id).e12*sig.at(id).e12;
	  tau.at(id).e13=sig.at(id).e13*sig.at(id).e13;

	  tau.at(id).e22=sig.at(id).e22*sig.at(id).e22;
	  tau.at(id).e23=sig.at(id).e23*sig.at(id).e23;
	  tau.at(id).e33=sig.at(id).e33*sig.at(id).e33;
	 	  
	  double detTau=(tau.at(id).e11*tau.at(id).e22*tau.at(id).e33)-
	    (tau.at(id).e11*tau.at(id).e23*tau.at(id).e23)-
	    (tau.at(id).e12*tau.at(id).e12*tau.at(id).e33)+
	    (tau.at(id).e12*tau.at(id).e23*tau.at(id).e13)+
	    (tau.at(id).e13*tau.at(id).e12*tau.at(id).e23)-
	    (tau.at(id).e13*tau.at(id).e22*tau.at(id).e13);
	 	  
	  lam.at(id).e11=( (tau.at(id).e22*tau.at(id).e33)-(tau.at(id).e23*tau.at(id).e23) )/detTau;
	  lam.at(id).e12=( (tau.at(id).e13*tau.at(id).e23)-(tau.at(id).e12*tau.at(id).e33) )/detTau;
	  lam.at(id).e13=( (tau.at(id).e12*tau.at(id).e23)-(tau.at(id).e13*tau.at(id).e22) )/detTau;
	  lam.at(id).e21=( (tau.at(id).e23*tau.at(id).e13)-(tau.at(id).e12*tau.at(id).e33) )/detTau;
	  lam.at(id).e22=( (tau.at(id).e11*tau.at(id).e33)-(tau.at(id).e13*tau.at(id).e13) )/detTau;
	  lam.at(id).e23=( (tau.at(id).e13*tau.at(id).e12)-(tau.at(id).e11*tau.at(id).e23) )/detTau;
	  lam.at(id).e31=( (tau.at(id).e12*tau.at(id).e23)-(tau.at(id).e22*tau.at(id).e13) )/detTau;
	  lam.at(id).e32=( (tau.at(id).e12*tau.at(id).e13)-(tau.at(id).e11*tau.at(id).e23) )/detTau;
	  lam.at(id).e33=( (tau.at(id).e11*tau.at(id).e22)-(tau.at(id).e12*tau.at(id).e12) )/detTau;


	}
    //    writeFile(sig,"sigma.dat");
    break;
  default:
    std::cerr<<"Cannot Create Sigmas"<<std::endl;
  }
  std::ofstream taus;
  taus.open("../bw/taus.dat");
  for(int k=0;k<nz;k++){
      for(int j=0;j<ny;j++){
          for(int i=0;i<nx;i++){
              int id=k*ny*nx + j*nx + i;            
              double tau11=tau.at(id).e11;
              double tau12=tau.at(id).e12;
              double tau13=tau.at(id).e13;
              double tau22=tau.at(id).e22;
              double tau23=tau.at(id).e23;
              double tau33=tau.at(id).e33;
              taus<<i<<"   "<<j<<"   "<<k<<"   "<<tau11<<"  "<<tau12<<"  "<<tau13<<"  "<<tau22<<"  "<<tau23<<"  "<<tau33<<std::endl;
          }
      }
  }
  taus.close();
  // writeFile(sig,"sigma.dat");
  for(int i=0; i<nz*ny*nx; i++)
  {
//     std::cout<<tau.at(i).e11<<"  "<<tau.at(i).e11<<std::endl;
  }
  createTauGrads();
  //delete turb;
}
  
  
void eulerian::createSigmaAndEpsQUIC(){
        std::cout<<"in create sigma eps QUIC"<<std::endl;

  CoEps.resize(nx*ny*nz);
  sig.resize(nx*ny*nz);
  tau.resize(nx*ny*nz);
  lam.resize(nx*ny*nz);

  std::ifstream QUICTurbField;
  QUICTurbField.open("../bw/QP_turbfield.dat");
  
  if(!QUICTurbField.is_open()){
    std::cerr<<"Turbulence input File open error"<<std::endl;
    exit(1);
  }
  std::string header;
  
   QUICTurbField>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header
		>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header
		>>header>>header>>header>>header>>header>>header>>header>>header>>header
		>>header>>header>>header>>header>>header>>header>>header>>header>>header
		>>header>>header>>header;


  double quicIndex;
  

  
  for(int k=0;k<nz;++k)
    for(int j=0;j<ny;++j)
      for(int i=0; i<nx;++i){
	
	int id=k*ny*nx + j*nx + i;
	double ustarQUIC=0.0;
	
	if(k==0){
	  sig.at(id).e11 = 0.0;
	  sig.at(id).e12 = 0.0;
	  sig.at(id).e13 = 0.0;
	  sig.at(id).e22 = 0.0;// * vertShear;
	  sig.at(id).e23 = 0.0;
	  sig.at(id).e33 = 0.0;// * vertShear;
	}
	else{
	  QUICTurbField>>quicIndex>>quicIndex>>quicIndex;

	  QUICTurbField>>ustarQUIC;
	  ustarQUIC=ustarQUIC/2.5;
	  double extra;
	  QUICTurbField>>extra>>extra>>extra>>extra>>extra>>extra>>extra>>extra;
	}
	  
	sig.at(id).e11 = 2.5 * ustarQUIC;
	sig.at(id).e12 = 0.0;
	sig.at(id).e13 = ustarQUIC;
	sig.at(id).e22 = 2.0 * ustarQUIC;
	sig.at(id).e23 = 0.0;
	sig.at(id).e33 = 1.3 * ustarQUIC;
	
	
	
	CoEps.at(id)=5.7*pow(ustarQUIC,3.0) / (vonKar*(zInMeters.at(k)+zo));
	
	tau.at(id).e11=sig.at(id).e11*sig.at(id).e11;
	tau.at(id).e12=sig.at(id).e12*sig.at(id).e12;
	tau.at(id).e13=sig.at(id).e13*sig.at(id).e13;
	
	tau.at(id).e22=sig.at(id).e22*sig.at(id).e22;
	tau.at(id).e23=sig.at(id).e23*sig.at(id).e23;
	tau.at(id).e33=sig.at(id).e33*sig.at(id).e33;
	
	double detTau=(tau.at(id).e11*tau.at(id).e22*tau.at(id).e33)-
	  (tau.at(id).e11*tau.at(id).e23*tau.at(id).e23)-
	  (tau.at(id).e12*tau.at(id).e12*tau.at(id).e33)+
	  (tau.at(id).e12*tau.at(id).e23*tau.at(id).e13)+
	  (tau.at(id).e13*tau.at(id).e12*tau.at(id).e23)-
	  (tau.at(id).e13*tau.at(id).e22*tau.at(id).e13);
	
	lam.at(id).e11=( (tau.at(id).e22*tau.at(id).e33)-(tau.at(id).e23*tau.at(id).e23) )/detTau;
	lam.at(id).e12=( (tau.at(id).e13*tau.at(id).e23)-(tau.at(id).e12*tau.at(id).e33) )/detTau;
	lam.at(id).e13=( (tau.at(id).e12*tau.at(id).e23)-(tau.at(id).e13*tau.at(id).e22) )/detTau;
	lam.at(id).e21=( (tau.at(id).e23*tau.at(id).e13)-(tau.at(id).e12*tau.at(id).e33) )/detTau;
	lam.at(id).e22=( (tau.at(id).e11*tau.at(id).e33)-(tau.at(id).e13*tau.at(id).e13) )/detTau;
	lam.at(id).e23=( (tau.at(id).e13*tau.at(id).e12)-(tau.at(id).e11*tau.at(id).e23) )/detTau;
	lam.at(id).e31=( (tau.at(id).e12*tau.at(id).e23)-(tau.at(id).e22*tau.at(id).e13) )/detTau;
	lam.at(id).e32=( (tau.at(id).e12*tau.at(id).e13)-(tau.at(id).e11*tau.at(id).e23) )/detTau;
	lam.at(id).e33=( (tau.at(id).e11*tau.at(id).e22)-(tau.at(id).e12*tau.at(id).e12) )/detTau;
      }
  // writeFile(sig,"sigma.dat");
  createTauGrads();
}

 
void eulerian::createSigmaAndEpsQUICFull(){
        std::cout<<"in create sigma eps QUIC"<<std::endl;

  CoEps.resize(nx*ny*nz);
  sig.resize(nx*ny*nz);
  tau.resize(nx*ny*nz);
  lam.resize(nx*ny*nz);

  std::ifstream QUICTurbField;
  QUICTurbField.open("../bw/QP_turbfield.dat");
  
  if(!QUICTurbField.is_open()){
    std::cerr<<"Turbulence input File open error"<<std::endl;
    exit(1);
  }
  std::string header;
  
   QUICTurbField>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header
		>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header
		>>header>>header>>header>>header>>header>>header>>header>>header>>header
		>>header>>header>>header>>header>>header>>header>>header>>header>>header
		>>header>>header>>header;


  double quicIndex;
  

  
  for(int k=0;k<nz;++k)
    for(int j=0;j<ny;++j)
      for(int i=0; i<nx;++i){
	
	int id=k*ny*nx + j*nx + i;
	double ustarQUIC=0.0;
	
	if(k==0){
	  sig.at(id).e11 = 0.0;
	  sig.at(id).e12 = 0.0;
	  sig.at(id).e13 = 0.0;
	  sig.at(id).e22 = 0.0;// * vertShear;
	  sig.at(id).e23 = 0.0;
	  sig.at(id).e33 = 0.0;// * vertShear;
	}
	else{
	  QUICTurbField>>quicIndex>>quicIndex>>quicIndex;

	  QUICTurbField>>sig.at(id).e11>>sig.at(id).e22>>sig.at(id).e33;
          double Lz,extra;
          QUICTurbField>>Lz;
          
          QUICTurbField>>extra>>extra;
          QUICTurbField>>sig.at(id).e12>>sig.at(id).e13>>sig.at(id).e23;

          ustarQUIC=sig.at(id).e11/2.5;

          //          CoEps.at(id)=5.7*pow(ustarQUIC,3.0) / (vonKar*(zInMeters.at(k)+zo));
          CoEps.at(id)=5.7*pow(ustarQUIC,3.0) / (Lz);
          
          tau.at(id).e11=sig.at(id).e11*sig.at(id).e11;
          tau.at(id).e12=sig.at(id).e12*sig.at(id).e12;
          tau.at(id).e13=sig.at(id).e13*sig.at(id).e13;
          
          tau.at(id).e22=sig.at(id).e22*sig.at(id).e22;
          tau.at(id).e23=sig.at(id).e23*sig.at(id).e23;
          tau.at(id).e33=sig.at(id).e33*sig.at(id).e33;

          if(tau.at(id).e11>1.)tau.at(id).e11=1.0;
          if(tau.at(id).e22>1.)tau.at(id).e11=1.0;
          if(tau.at(id).e33>1.)tau.at(id).e11=1.0;
          if(tau.at(id).e12>1.)tau.at(id).e11=1.0;
          if(tau.at(id).e13>1.)tau.at(id).e11=1.0;
          if(tau.at(id).e23>1.)tau.at(id).e11=1.0;
                                            
          //std::cout<<sig.at(id).e11<<"  "<<sig.at(id).e22<<"   "<<sig.at(id).e33<<"  "<<Lz<<std::endl;
          //std::cout<<sig.at(id).e12<<"  "<<sig.at(id).e13<<"   "<<sig.at(id).e23<<std::endl;
          //getchar();
	}

	  
	
	double detTau=(tau.at(id).e11*tau.at(id).e22*tau.at(id).e33)-
	  (tau.at(id).e11*tau.at(id).e23*tau.at(id).e23)-
	  (tau.at(id).e12*tau.at(id).e12*tau.at(id).e33)+
	  (tau.at(id).e12*tau.at(id).e23*tau.at(id).e13)+
	  (tau.at(id).e13*tau.at(id).e12*tau.at(id).e23)-
	  (tau.at(id).e13*tau.at(id).e22*tau.at(id).e13);
	
	lam.at(id).e11=( (tau.at(id).e22*tau.at(id).e33)-(tau.at(id).e23*tau.at(id).e23) )/detTau;
	lam.at(id).e12=( (tau.at(id).e13*tau.at(id).e23)-(tau.at(id).e12*tau.at(id).e33) )/detTau;
	lam.at(id).e13=( (tau.at(id).e12*tau.at(id).e23)-(tau.at(id).e13*tau.at(id).e22) )/detTau;
	lam.at(id).e21=( (tau.at(id).e23*tau.at(id).e13)-(tau.at(id).e12*tau.at(id).e33) )/detTau;
	lam.at(id).e22=( (tau.at(id).e11*tau.at(id).e33)-(tau.at(id).e13*tau.at(id).e13) )/detTau;
	lam.at(id).e23=( (tau.at(id).e13*tau.at(id).e12)-(tau.at(id).e11*tau.at(id).e23) )/detTau;
	lam.at(id).e31=( (tau.at(id).e12*tau.at(id).e23)-(tau.at(id).e22*tau.at(id).e13) )/detTau;
	lam.at(id).e32=( (tau.at(id).e12*tau.at(id).e13)-(tau.at(id).e11*tau.at(id).e23) )/detTau;
	lam.at(id).e33=( (tau.at(id).e11*tau.at(id).e22)-(tau.at(id).e12*tau.at(id).e12) )/detTau;
      }
  // writeFile(sig,"sigma.dat");
  createTauGrads();
}

  
void eulerian::createUstar(){
        std::cout<<"in crete ustar"<<std::endl;
  ustar.resize(nx*ny*nz);
  dudz.resize(nx*ny*nz);
  
  for(int k=0; k<nz;++k)
    for(int j=0;j<ny;++j)
      for(int i=0;i<nx;++i){
	int id=k*ny*nx+j*nx+i;

	int idZm1=(k-1)*ny*nx+j*nx+i;
	int idZp1=(k+1)*ny*nx+j*nx+i;
	
	if(k==0){//ground
	  dudz.at(id)=0.0;
	  ustar.at(id)=0.0;
	}
	else if(k==1){//just above ground, log law
	  dudz.at(id) =windVec.at(id).u / (zInMeters.at(k)*(log(zInMeters.at(k)/zo)));//windVec.at(id).u/(0.5*dz);//
	  ustar.at(id)= vonKar*zInMeters.at(k)*dudz.at(id);
	}
	else if(k==nz-1){
	  dudz.at(id)=dudz.at(idZm1);
	  ustar.at(id)=vonKar*zInMeters.at(k)*dudz.at(idZm1);
	}
	else{
	  dudz.at(id)= ( windVec.at(idZp1).u-windVec.at(idZm1).u ) / (2.0*dz);
	  ustar.at(id)= vonKar*zInMeters.at(k) * dudz.at(id);
	}
	if(i==1 && j==1 && 0){
	  std::cout<<"dudz: "<<dudz.at(id)<<std::endl;
	  std::cout<<"ustar: "<<ustar.at(id)<<std::endl;
	  std::cout<<"U: "<<windVec.at(id).u<<std::endl;
	  std::cout<<"Dis: "<<zInMeters.at(k)<<std::endl;
	  std::cout<<"zo: "<<zo<<std::endl;
	  std::cout<<"K:"<<vonKar<<"   "<<std::endl;
	  //getchar();
	}
      }
}

void eulerian::createTauGrads(){

  std::cout<<"eulerian::createTauGrads..."<<std::endl;

  taudx.resize(nx*ny*nz);
  taudy.resize(nx*ny*nz);
  taudz.resize(nx*ny*nz);
  //createA1Matrix();
  //return;

  std::ofstream taugradsdx;
  taugradsdx.open("../bw/taugradsdx.dat");

  std::ofstream taugradsdy;
  taugradsdy.open("../bw/taugradsdy.dat");

  std::ofstream taugradsdz;
  taugradsdz.open("../bw/taugradsdz.dat");


  for(int k=0; k<nz;++k){
    for(int j=0;j<ny;++j){
      for(int i=0;i<nx;++i){

	int id=k*ny*nx+j*nx+i;

	if(i>1 && i<nx-1 && j>1 && j<ny-1 && k>0 && k<nz-1 && CellType.at(id).c!=0){
	  int idXp1=k*ny*nx+j*nx+(i+1);
	  int idXm1=k*ny*nx+j*nx+(i-1);

	  int idXp2=k*ny*nx+j*nx+(i+2);
	  int idXm2=k*ny*nx+j*nx+(i-2);

	  int idYp1=k*ny*nx+(j+1)*nx+i;
	  int idYm1=k*ny*nx+(j-1)*nx+i;
	  int idZp1=(k+1)*ny*nx+j*nx+i;
	  int idZm1=(k-1)*ny*nx+j*nx+i;
	  int XXX=1;
	  int YYY=1;
	  int ZZZ=1;
	  
	  taudx.at(id).e11= 0.0;
	  taudx.at(id).e12= 0.0;
	  taudx.at(id).e13= 0.0;
	  taudx.at(id).e22= 0.0;
	  taudx.at(id).e23= 0.0;
	  taudx.at(id).e33= 0.0;
	  if((CellType.at(idXm1).c==0 && CellType.at(id).c!=0)||(CellType.at(idXp1).c==0 && CellType.at(id).c!=0)){ //check this condition, its NOT right!!!
              /*taudx.at(id).e11= tau.at(id).e11;
              taudx.at(id).e12= tau.at(id).e11;
              taudx.at(id).e13= tau.at(id).e11;
              taudx.at(id).e22= tau.at(id).e11;
              taudx.at(id).e23= tau.at(id).e11;
              taudx.at(id).e33= tau.at(id).e11;*/
	  }
	  else if(XXX){
	    //central frist order difference 
              taudx.at(id).e11= ( tau.at(idXp1).e11-tau.at(idXm1).e11 ) / 2.0;
              taudx.at(id).e12= ( tau.at(idXp1).e12-tau.at(idXm1).e12 ) / 2.0;
              taudx.at(id).e13= ( tau.at(idXp1).e13-tau.at(idXm1).e13 ) / 2.0;
              taudx.at(id).e22= ( tau.at(idXp1).e22-tau.at(idXm1).e22 ) / 2.0;
              taudx.at(id).e23= ( tau.at(idXp1).e23-tau.at(idXm1).e23 ) / 2.0;
              taudx.at(id).e33= ( tau.at(idXp1).e33-tau.at(idXm1).e33 ) / 2.0;
	  }
          
	  
	  taudy.at(id).e11= 0.0;
	  taudy.at(id).e12= 0.0;
	  taudy.at(id).e13= 0.0;
	  taudy.at(id).e22= 0.0;
	  taudy.at(id).e23= 0.0;
	  taudy.at(id).e33= 0.0;
	  if( (CellType.at(idYm1).c==0 && CellType.at(id).c!=0) ||(CellType.at(idYp1).c==0 && CellType.at(id).c!=0)){
              /*taudy.at(id).e11= tau.at(id).e11;
              taudy.at(id).e12= tau.at(id).e12;
              taudy.at(id).e13= tau.at(id).e13;
              taudy.at(id).e22= tau.at(id).e22;
              taudy.at(id).e23= tau.at(id).e23;
              taudy.at(id).e33= tau.at(id).e33;*/
	  }
	  else if(YYY){
              taudy.at(id).e11= ( tau.at(idYp1).e11-tau.at(idYm1).e11 ) / 2.0;
              taudy.at(id).e12= ( tau.at(idYp1).e12-tau.at(idYm1).e12 ) / 2.0;
              taudy.at(id).e13= ( tau.at(idYp1).e13-tau.at(idYm1).e13 ) / 2.0;
              taudy.at(id).e22= ( tau.at(idYp1).e22-tau.at(idYm1).e22 ) / 2.0;
              taudy.at(id).e23= ( tau.at(idYp1).e23-tau.at(idYm1).e23 ) / 2.0;
              taudy.at(id).e33= ( tau.at(idYp1).e33-tau.at(idYm1).e33 ) / 2.0;
	  }
          
	  taudz.at(id).e11= 0.0;
	  taudz.at(id).e12= 0.0;
	  taudz.at(id).e13= 0.0;
	  taudz.at(id).e22= 0.0;
	  taudz.at(id).e23= 0.0;
	  taudz.at(id).e33= 0.0;
	  
	  if(CellType.at(idZm1).c==0 && CellType.at(id).c!=0){
              taudz.at(id).e11=tau.at(id).e11;
              taudz.at(id).e12=tau.at(id).e12;
              taudz.at(id).e13=tau.at(id).e13;
              taudz.at(id).e22=tau.at(id).e22;
              taudz.at(id).e23=tau.at(id).e23;
              taudz.at(id).e33=tau.at(id).e33;
	  }
	  else if(ZZZ){
              taudz.at(id).e11= ( tau.at(idZp1).e11-tau.at(idZm1).e11 ) / 2.0;
              taudz.at(id).e12= ( tau.at(idZp1).e12-tau.at(idZm1).e12 ) / 2.0;
              taudz.at(id).e13= ( tau.at(idZp1).e13-tau.at(idZm1).e13 ) / 2.0;
              taudz.at(id).e22= ( tau.at(idZp1).e22-tau.at(idZm1).e22 ) / 2.0;
              taudz.at(id).e23= ( tau.at(idZp1).e23-tau.at(idZm1).e23 ) / 2.0;
              taudz.at(id).e33= ( tau.at(idZp1).e33-tau.at(idZm1).e33 ) / 2.0;
	  }
	}//if loop for domain
        
        /*	taugradsdx<<i<<"  "<<j<<"  "<<k<<"  "<<taudx.at(id).e11<<"  "<<taudx.at(id).e22<<"  "<<taudx.at(id).e33<<"  "<<taudx.at(id).e12<<"  "<<taudx.at(id).e13<<"  "<<taudx.at(id).e23<<std::endl;

	taugradsdy<<i<<"  "<<j<<"  "<<k<<"  "<<taudy.at(id).e11<<"  "<<taudy.at(id).e22<<"  "<<taudy.at(id).e33<<"  "<<taudy.at(id).e12<<"  "<<taudy.at(id).e13<<"  "<<taudy.at(id).e23<<std::endl;

	taugradsdz<<i<<"  "<<j<<"  "<<k<<"  "<<taudz.at(id).e11<<"  "<<taudz.at(id).e22<<"  "<<taudz.at(id).e33<<"  "<<taudz.at(id).e12<<"  "<<taudz.at(id).e13<<"  "<<taudz.at(id).e23<<std::endl;
	*/


	
	
	/*	if(i>1 && j>1 && i<nx-2 && j<ny-2 && k>0 && CellType.at(id).c!=0){ //all cells except boundary cells
	  if(k==1){//just above ground

	    taudz.at(id).e11=0.0;//0.5*(tau.at(idZp2).e11-tau.at(id).e11)/dz;
	    taudz.at(id).e12=0.0;//0.5*(tau.at(idZp2).e12-tau.at(id).e12)/dz;
	    taudz.at(id).e13=0.0;//0.5*(tau.at(idZp2).e13-tau.at(id).e13)/dz;
	    taudz.at(id).e22=0.0;//0.5*(tau.at(idZp2).e22-tau.at(id).e22)/dz;
	    taudz.at(id).e23=0.0;//0.5*(tau.at(idZp2).e23-tau.at(id).e23)/dz;
	    taudz.at(id).e33=0.0;//0.5*(tau.at(idZp2).e33-tau.at(id).e33)/dz;
	    
	    
	    taudx.at(id).e11= 0.0;//( tau.at(idXp1).e11-tau.at(idXm1).e11 ) / 2.0;
	    taudx.at(id).e12= 0.0;//( tau.at(idXp1).e12-tau.at(idXm1).e12 ) / 2.0;
	    taudx.at(id).e13= 0.0;//( tau.at(idXp1).e13-tau.at(idXm1).e13 ) / 2.0;
	    taudx.at(id).e22= 0.0;//( tau.at(idXp1).e22-tau.at(idXm1).e22 ) / 2.0;
	    taudx.at(id).e23= 0.0;//( tau.at(idXp1).e23-tau.at(idXm1).e23 ) / 2.0;
	    taudx.at(id).e33= 0.0;//( tau.at(idXp1).e33-tau.at(idXm1).e33 ) / 2.0;
	    
	    taudy.at(id).e11= 0.0;// ( tau.at(idYp1).e11-tau.at(idYm1).e11 ) / 2.0;
	    taudy.at(id).e12= 0.0;//( tau.at(idYp1).e12-tau.at(idYm1).e12 ) / 2.0;
	    taudy.at(id).e13= 0.0;// ( tau.at(idYp1).e13-tau.at(idYm1).e13 ) / 2.0;
	    taudy.at(id).e22= 0.0;// ( tau.at(idYp1).e22-tau.at(idYm1).e22 ) / 2.0;
	    taudy.at(id).e23= 0.0;// ( tau.at(idYp1).e23-tau.at(idYm1).e23 ) / 2.0;
	    taudy.at(id).e33= 0.0;// ( tau.at(idYp1).e33-tau.at(idYm1).e33 ) / 2.0;
	    
	  }
	  else if(k==nz-1){

	    taudz.at(id).e11= taudz.at(idZm1).e11;
	    taudz.at(id).e12= taudz.at(idZm1).e12;
	    taudz.at(id).e13= taudz.at(idZm1).e13;
	    taudz.at(id).e22= taudz.at(idZm1).e22;
	    taudz.at(id).e23= taudz.at(idZm1).e23;
	    taudz.at(id).e33= taudz.at(idZm1).e33;
	    
	    taudx.at(id).e11= ( tau.at(idXp1).e11-tau.at(idXm1).e11 ) / 2.0;
	    taudx.at(id).e12= ( tau.at(idXp1).e12-tau.at(idXm1).e12 ) / 2.0;
	    taudx.at(id).e13= ( tau.at(idXp1).e13-tau.at(idXm1).e13 ) / 2.0;
	    taudx.at(id).e22= ( tau.at(idXp1).e22-tau.at(idXm1).e22 ) / 2.0;
	    taudx.at(id).e23= ( tau.at(idXp1).e23-tau.at(idXm1).e23 ) / 2.0;
	    taudx.at(id).e33= ( tau.at(idXp1).e33-tau.at(idXm1).e33 ) / 2.0;
	    
	    taudy.at(id).e11= ( tau.at(idYp1).e11-tau.at(idYm1).e11 ) / 2.0;
	    taudy.at(id).e12= ( tau.at(idYp1).e12-tau.at(idYm1).e12 ) / 2.0;
	    taudy.at(id).e13= ( tau.at(idYp1).e13-tau.at(idYm1).e13 ) / 2.0;
	    taudy.at(id).e22= ( tau.at(idYp1).e22-tau.at(idYm1).e22 ) / 2.0;
	    taudy.at(id).e23= ( tau.at(idYp1).e23-tau.at(idYm1).e23 ) / 2.0;
	    taudy.at(id).e33= ( tau.at(idYp1).e33-tau.at(idYm1).e33 ) / 2.0;
	    
	  }
	  else{

	    	    taudx.at(id).e11= ( tau.at(idXp1).e11-tau.at(idXm1).e11 ) / 2.0;
	    taudx.at(id).e12= ( tau.at(idXp1).e12-tau.at(idXm1).e12 ) / 2.0;
	    taudx.at(id).e13= ( tau.at(idXp1).e13-tau.at(idXm1).e13 ) / 2.0;
	    taudx.at(id).e22= ( tau.at(idXp1).e22-tau.at(idXm1).e22 ) / 2.0;
	    taudx.at(id).e23= ( tau.at(idXp1).e23-tau.at(idXm1).e23 ) / 2.0;
	    taudx.at(id).e33= ( tau.at(idXp1).e33-tau.at(idXm1).e33 ) / 2.0;
	    
	    taudy.at(id).e11= ( tau.at(idYp1).e11-tau.at(idYm1).e11 ) / 2.0;
	    taudy.at(id).e12= ( tau.at(idYp1).e12-tau.at(idYm1).e12 ) / 2.0;
	    taudy.at(id).e13= ( tau.at(idYp1).e13-tau.at(idYm1).e13 ) / 2.0;
	    taudy.at(id).e22= ( tau.at(idYp1).e22-tau.at(idYm1).e22 ) / 2.0;
	    taudy.at(id).e23= ( tau.at(idYp1).e23-tau.at(idYm1).e23 ) / 2.0;
	    taudy.at(id).e33= ( tau.at(idYp1).e33-tau.at(idYm1).e33 ) / 2.0;
	    
	    taudz.at(id).e11= ( tau.at(idZp1).e11-tau.at(idZm1).e11 ) / 2.0;
	    taudz.at(id).e12= ( tau.at(idZp1).e12-tau.at(idZm1).e12 ) / 2.0;
	    taudz.at(id).e13= ( tau.at(idZp1).e13-tau.at(idZm1).e13 ) / 2.0;
	    taudz.at(id).e22= ( tau.at(idZp1).e22-tau.at(idZm1).e22 ) / 2.0;
	    taudz.at(id).e23= ( tau.at(idZp1).e23-tau.at(idZm1).e23 ) / 2.0;
	    taudz.at(id).e33= ( tau.at(idZp1).e33-tau.at(idZm1).e33 ) / 2.0;
	  }
	}
	else{//boundary cells

	  taudz.at(id).e11=0.0;
	  taudz.at(id).e12=0.0;
	  taudz.at(id).e13=0.0;
	  taudz.at(id).e22=0.0;
	  taudz.at(id).e23=0.0;
	  taudz.at(id).e33=0.0;
	  
	  
	  taudx.at(id).e11=0.0;
	  taudx.at(id).e12=0.0;
	  taudx.at(id).e13=0.0;
	  taudx.at(id).e22=0.0;
	  taudx.at(id).e23=0.0;
	  taudx.at(id).e33=0.0;
	  
	  taudy.at(id).e11=0.0;
	  taudy.at(id).e12=0.0;
	  taudy.at(id).e13=0.0;
	  taudy.at(id).e22=0.0;
	  taudy.at(id).e23=0.0;
	  taudy.at(id).e33=0.0;
	  }*/
      }
    }
  }
  
  createA1Matrix();
}

void eulerian::createA1Matrix(){
        std::cout<<"in createA1"<<std::endl;
  std::cout<<"INside A1 creat"<<std::endl;
  std::ofstream outfile;
  outfile.open("../bw/out.dat");

  std::ofstream CondNum_A1;
  CondNum_A1.open("../bw/CondNum_A1.dat");

  std::ofstream Det_A1;
  Det_A1.open("../bw/Det_A1.dat");

  eigVal.resize(nx*ny*nz);
  eigVec.resize(nx*ny*nz);
  eigVecInv.resize(nx*ny*nz);
  ka0.resize(nx*ny*nz);

  g2nd.resize(nx*ny*nz);
  
  double cond_A1=0.0;
  double det_A1=0.0;
  int number=0;
  int flagnum=0;
  
  
  for(int k=1;k<nz-1;k++)
    for(int j=1;j<ny-1;j++)
      for(int i=1;i<nx-1;i++){
	
	int id=k*ny*nx+j*nx+i;
	int idXp1=k*ny*nx+j*nx+(i+1);
	int idXm1=k*ny*nx+j*nx+(i-1);
	int idYp1=k*ny*nx+(j+1)*nx+i;
	int idYm1=k*ny*nx+(j-1)*nx+i;
	int idZp1=(k+1)*ny*nx+j*nx+i;
	int idZm1=(k-1)*ny*nx+j*nx+i;
	cond_A1=0.0;
	det_A1=0.0;
	
	if(CellType.at(id).c!=0){
	  //std::cout<<i<<"   "<<j<<"   "<<k<<std::endl;
	  
	  
	  double A1_1e11= -0.5*CoEps.at(id)*lam.at(id).e11;
	  double A1_1e12= -0.5*CoEps.at(id)*lam.at(id).e12;
	  double A1_1e13= -0.5*CoEps.at(id)*lam.at(id).e13;
	  double A1_1e21= -0.5*CoEps.at(id)*lam.at(id).e21;
	  double A1_1e22= -0.5*CoEps.at(id)*lam.at(id).e22;
	  double A1_1e23= -0.5*CoEps.at(id)*lam.at(id).e23;
	  double A1_1e31= -0.5*CoEps.at(id)*lam.at(id).e31;
	  double A1_1e32= -0.5*CoEps.at(id)*lam.at(id).e32;
	  double A1_1e33= -0.5*CoEps.at(id)*lam.at(id).e33;
	  
	  double A1_2e11= 0.5*lam.at(id).e11*taudx.at(id).e11*windVec.at(id).u;
	  double A1_2e12= 0.5*lam.at(id).e11*taudx.at(id).e12*windVec.at(id).u;
	  double A1_2e13= 0.5*lam.at(id).e11*taudx.at(id).e13*windVec.at(id).u;
	  double A1_2e21= 0.5*lam.at(id).e12*taudx.at(id).e11*windVec.at(id).u;
	  double A1_2e22= 0.5*lam.at(id).e12*taudx.at(id).e12*windVec.at(id).u;
	  double A1_2e23= 0.5*lam.at(id).e12*taudx.at(id).e13*windVec.at(id).u;
	  double A1_2e31= 0.5*lam.at(id).e13*taudx.at(id).e11*windVec.at(id).u;
	  double A1_2e32= 0.5*lam.at(id).e13*taudx.at(id).e12*windVec.at(id).u;
	  double A1_2e33= 0.5*lam.at(id).e13*taudx.at(id).e13*windVec.at(id).u;
	  
	  double A1_3e11= 0.5*lam.at(id).e21*taudx.at(id).e12*windVec.at(id).u;
	  double A1_3e12= 0.5*lam.at(id).e21*taudx.at(id).e22*windVec.at(id).u;
	  double A1_3e13= 0.5*lam.at(id).e21*taudx.at(id).e23*windVec.at(id).u;
	  double A1_3e21= 0.5*lam.at(id).e22*taudx.at(id).e12*windVec.at(id).u;
	  double A1_3e22= 0.5*lam.at(id).e22*taudx.at(id).e22*windVec.at(id).u;
	  double A1_3e23= 0.5*lam.at(id).e22*taudx.at(id).e23*windVec.at(id).u;
	  double A1_3e31= 0.5*lam.at(id).e23*taudx.at(id).e12*windVec.at(id).u;
	  double A1_3e32= 0.5*lam.at(id).e23*taudx.at(id).e22*windVec.at(id).u;
	  double A1_3e33= 0.5*lam.at(id).e23*taudx.at(id).e23*windVec.at(id).u;
	  
	  double A1_4e11= 0.5*lam.at(id).e31*taudx.at(id).e13*windVec.at(id).u;
	  double A1_4e12= 0.5*lam.at(id).e31*taudx.at(id).e23*windVec.at(id).u;
	  double A1_4e13= 0.5*lam.at(id).e31*taudx.at(id).e33*windVec.at(id).u;
	  double A1_4e21= 0.5*lam.at(id).e32*taudx.at(id).e13*windVec.at(id).u;
	  double A1_4e22= 0.5*lam.at(id).e32*taudx.at(id).e23*windVec.at(id).u;
	  double A1_4e23= 0.5*lam.at(id).e32*taudx.at(id).e33*windVec.at(id).u;
	  double A1_4e31= 0.5*lam.at(id).e33*taudx.at(id).e13*windVec.at(id).u;
	  double A1_4e32= 0.5*lam.at(id).e33*taudx.at(id).e23*windVec.at(id).u;
	  double A1_4e33= 0.5*lam.at(id).e33*taudx.at(id).e33*windVec.at(id).u;
	  
	  double A1_5e11= 0.5*lam.at(id).e11*taudy.at(id).e11*windVec.at(id).v;
	  double A1_5e12= 0.5*lam.at(id).e11*taudy.at(id).e12*windVec.at(id).v;
	  double A1_5e13= 0.5*lam.at(id).e11*taudy.at(id).e13*windVec.at(id).v;
	  double A1_5e21= 0.5*lam.at(id).e12*taudy.at(id).e11*windVec.at(id).v;
	  double A1_5e22= 0.5*lam.at(id).e12*taudy.at(id).e12*windVec.at(id).v;
	  double A1_5e23= 0.5*lam.at(id).e12*taudy.at(id).e13*windVec.at(id).v;
	  double A1_5e31= 0.5*lam.at(id).e13*taudy.at(id).e11*windVec.at(id).v;
	  double A1_5e32= 0.5*lam.at(id).e13*taudy.at(id).e12*windVec.at(id).v;
	  double A1_5e33= 0.5*lam.at(id).e13*taudy.at(id).e13*windVec.at(id).v;
	  
	  double A1_6e11= 0.5*lam.at(id).e21*taudy.at(id).e12*windVec.at(id).v;
	  double A1_6e12= 0.5*lam.at(id).e21*taudy.at(id).e22*windVec.at(id).v;
	  double A1_6e13= 0.5*lam.at(id).e21*taudy.at(id).e23*windVec.at(id).v;
	  double A1_6e21= 0.5*lam.at(id).e22*taudy.at(id).e12*windVec.at(id).v;
	  double A1_6e22= 0.5*lam.at(id).e22*taudy.at(id).e22*windVec.at(id).v;
	  double A1_6e23= 0.5*lam.at(id).e22*taudy.at(id).e23*windVec.at(id).v;
	  double A1_6e31= 0.5*lam.at(id).e23*taudy.at(id).e12*windVec.at(id).v;
	  double A1_6e32= 0.5*lam.at(id).e23*taudy.at(id).e22*windVec.at(id).v;
	  double A1_6e33= 0.5*lam.at(id).e23*taudy.at(id).e23*windVec.at(id).v;
	  
	  double A1_7e11= 0.5*lam.at(id).e31*taudy.at(id).e13*windVec.at(id).v;
	  double A1_7e12= 0.5*lam.at(id).e31*taudy.at(id).e23*windVec.at(id).v;
	  double A1_7e13= 0.5*lam.at(id).e31*taudy.at(id).e33*windVec.at(id).v;
	  double A1_7e21= 0.5*lam.at(id).e32*taudy.at(id).e13*windVec.at(id).v;
	  double A1_7e22= 0.5*lam.at(id).e32*taudy.at(id).e23*windVec.at(id).v;
	  double A1_7e23= 0.5*lam.at(id).e32*taudy.at(id).e33*windVec.at(id).v;
	  double A1_7e31= 0.5*lam.at(id).e33*taudy.at(id).e13*windVec.at(id).v;
	  double A1_7e32= 0.5*lam.at(id).e33*taudy.at(id).e23*windVec.at(id).v;
	  double A1_7e33= 0.5*lam.at(id).e33*taudy.at(id).e33*windVec.at(id).v;
	  
	  
	  double A1_8e11= 0.5*lam.at(id).e11*taudz.at(id).e11*windVec.at(id).w;
	  double A1_8e12= 0.5*lam.at(id).e11*taudz.at(id).e12*windVec.at(id).w;
	  double A1_8e13= 0.5*lam.at(id).e11*taudz.at(id).e13*windVec.at(id).w;
	  double A1_8e21= 0.5*lam.at(id).e12*taudz.at(id).e11*windVec.at(id).w;
	  double A1_8e22= 0.5*lam.at(id).e12*taudz.at(id).e12*windVec.at(id).w;
	  double A1_8e23= 0.5*lam.at(id).e12*taudz.at(id).e13*windVec.at(id).w;
	  double A1_8e31= 0.5*lam.at(id).e13*taudz.at(id).e11*windVec.at(id).w;
	  double A1_8e32= 0.5*lam.at(id).e13*taudz.at(id).e12*windVec.at(id).w;
	  double A1_8e33= 0.5*lam.at(id).e13*taudz.at(id).e13*windVec.at(id).w;
	  
	  double A1_9e11= 0.5*lam.at(id).e21*taudz.at(id).e12*windVec.at(id).w;
	  double A1_9e12= 0.5*lam.at(id).e21*taudz.at(id).e22*windVec.at(id).w;
	  double A1_9e13= 0.5*lam.at(id).e21*taudz.at(id).e23*windVec.at(id).w;
	  double A1_9e21= 0.5*lam.at(id).e22*taudz.at(id).e12*windVec.at(id).w;
	  double A1_9e22= 0.5*lam.at(id).e22*taudz.at(id).e22*windVec.at(id).w;
	  double A1_9e23= 0.5*lam.at(id).e22*taudz.at(id).e23*windVec.at(id).w;
	  double A1_9e31= 0.5*lam.at(id).e23*taudz.at(id).e12*windVec.at(id).w;
	  double A1_9e32= 0.5*lam.at(id).e23*taudz.at(id).e22*windVec.at(id).w;
	  double A1_9e33= 0.5*lam.at(id).e23*taudz.at(id).e23*windVec.at(id).w;
	  
	  double A1_10e11= 0.5*lam.at(id).e31*taudz.at(id).e13*windVec.at(id).w;
	  double A1_10e12= 0.5*lam.at(id).e31*taudz.at(id).e23*windVec.at(id).w;
	  double A1_10e13= 0.5*lam.at(id).e31*taudz.at(id).e33*windVec.at(id).w;
	  double A1_10e21= 0.5*lam.at(id).e32*taudz.at(id).e13*windVec.at(id).w;
	  double A1_10e22= 0.5*lam.at(id).e32*taudz.at(id).e23*windVec.at(id).w;
	  double A1_10e23= 0.5*lam.at(id).e32*taudz.at(id).e33*windVec.at(id).w;
	  double A1_10e31= 0.5*lam.at(id).e33*taudz.at(id).e13*windVec.at(id).w;
	  double A1_10e32= 0.5*lam.at(id).e33*taudz.at(id).e23*windVec.at(id).w;
	  double A1_10e33= 0.5*lam.at(id).e33*taudz.at(id).e33*windVec.at(id).w;
	  
	  double A1e11= A1_1e11 + A1_2e11 + A1_3e11 + A1_4e11 +
	    A1_5e11 + A1_6e11 + A1_7e11 + A1_8e11 + A1_9e11 + A1_10e11;
	  
	  double A1e12= A1_1e12 + A1_2e12 + A1_3e12 + A1_4e12 +
	    A1_5e12 + A1_6e12 + A1_7e12 + A1_8e12 + A1_9e12 + A1_10e12;
	  
	  double A1e13= A1_1e13 + A1_2e13 + A1_3e13 + A1_4e13 +
	    A1_5e13 + A1_6e13 + A1_7e13 + A1_8e13 + A1_9e13 + A1_10e13;

	  double A1e21= A1_1e21 + A1_2e21 + A1_3e21 + A1_4e21 +
	    A1_5e21 + A1_6e21 + A1_7e21 + A1_8e21 + A1_9e21 + A1_10e21;
	  
	  double A1e22= A1_1e22 + A1_2e22 + A1_3e22 + A1_4e22 +
	    A1_5e22 + A1_6e22 + A1_7e22 + A1_8e22 + A1_9e22 + A1_10e22;
	  
	  double A1e23= A1_1e23 + A1_2e23 + A1_3e23 + A1_4e23 +
	    A1_5e23 + A1_6e23 + A1_7e23 + A1_8e23 + A1_9e23 + A1_10e23;
	  
	  double A1e31= A1_1e31 + A1_2e31 + A1_3e31 + A1_4e31 +
	    A1_5e31 + A1_6e31 + A1_7e31 + A1_8e31 + A1_9e31 + A1_10e31;
	  
	  double A1e32= A1_1e32 + A1_2e32 + A1_3e32 + A1_4e32 +
	    A1_5e32 + A1_6e32 + A1_7e32 + A1_8e32 + A1_9e32 + A1_10e32;
	  
	  double A1e33= A1_1e33 + A1_2e33 + A1_3e33 + A1_4e33 +
	    A1_5e33 + A1_6e33 + A1_7e33 + A1_8e33 + A1_9e33 + A1_10e33;

	  
	  bool imaginary=true;
          matrix9 mat9;
          while(imaginary){
              imaginary=false;

              mat9.e11=A1e11;
              mat9.e12=A1e12;
              mat9.e13=A1e13;
              mat9.e21=A1e21;
              mat9.e22=A1e22;
              mat9.e23=A1e23;
              mat9.e31=A1e31;
              mat9.e32=A1e32;
              mat9.e33=A1e33;
              cond_A1=matCondFro(mat9);
              det_A1=matrixDet(mat9);
              
              //For solving cubic equation (source http://en.wikipedia.org/wiki/Eigenvalue_algorithm
              //and http://www.1728.com/cubic2.htm)
              
              
              double a=-1; //ax^3+bx^2+cx+d=0
              double b=A1e11+A1e22+A1e33;
              
              double c=A1e12*A1e21 + A1e13*A1e31 + A1e23*A1e32
                  -A1e11*A1e22 - A1e11*A1e33 - A1e22*A1e33;
              
              double d= A1e11*A1e22*A1e33 - A1e11*A1e23*A1e32
                  -A1e21*A1e12*A1e33 + A1e21*A1e13*A1e32
                  +A1e31*A1e12*A1e23 - A1e31*A1e13*A1e22;
              
              //checking if the roots are real of imaginary
              double f=( (3.0*c/a)-((b*b)/(a*a)) ) / 3.0;
              double g=( ((2.0*b*b*b)/(a*a*a)) - ((9.0*b*c)/(a*a)) + (27.0*d/a) ) / 27.0;
              double h= (g*g/4.0) + (f*f*f/27.0);
              
              double tolP=1e-100;//tolerance on positive side (as h is double not an int, we cannot use equality logical operator)
              double tolN=-1e-100;//tolerance on negative side
              
              
              //Three cases
              if(h>1e-3){//1 real root, 2 imaginary roots
                  imaginary=true;
                  A1e12=0;
                  A1e13=0;
                  A1e21=0;
                  A1e23=0;
                  A1e31=0;
                  A1e32=0;
                  
             
                  
                  int iV=id%nx;
                  int jV=(id/nx)%ny;
                  int kV=(id/(nx*ny))%nz;
                  std::cerr<<"Imaginary roots ....exiting as h ="<<h<<std::endl;
                  std::cout<< "For equatio ax^3 + bx^2 + cx + d=0"<<std::endl;
                  std::cout<<"a :"<<a<<std::endl;
                  std::cout<<"b :"<<b<<std::endl;
                  std::cout<<"c :"<<c<<std::endl;
                  std::cout<<"d :"<<d<<std::endl;
                  std::cout<<"The original matrix is..."<<std::endl;
                  display(mat9);
                  std::cout<<"The d(tau)/dx  matrix is..."<<std::endl;
                  display(taudx.at(id));
                  std::cout<<"The d(tau)/dy  matrix is..."<<std::endl;
                  display(taudy.at(id));
                  std::cout<<"The d(tau)/dz  matrix is..."<<std::endl;
                  display(taudz.at(id));
                  
                  std::cout<<"The tau  matrix is..."<<std::endl;
                  display(tau.at(id));
                  
                  std::cout<<"The lamda  matrix is..."<<std::endl;
                  display(lam.at(id));
                  std::cout<<"The CoEps is..."<<std::endl;
                  std::cout<<CoEps.at(id)<<std::endl;
                  
                  std::cout<<"indicies at which this happend are (i,j,k) :"<<iV<<"   "<<jV<<"   "<<kV<<std::endl;
                  //	    exit(1);
              }
              else if(h<=tolP && h>=tolN && g<=tolP && g>=tolN && f<=tolP && f>=tolN){// All roots are real and equal
                  eigVal.at(id).e11=pow(d/a,1.0/3.0);
                  eigVal.at(id).e22=eigVal.at(id).e11;
                  eigVal.at(id).e33=eigVal.at(id).e11;
              }
              else{ //real roots
                  double ii=sqrt( (g*g/4.0)-h );
                  double jj=pow(ii,1.0/3.0);
                  double kk=acos( -(g/(2.0*ii)) );
                  double L=-1.0*jj;
                  double M=cos(kk/3.0);
                  double N=sqrt(3.0)*sin(kk/3.0);
                  double P=-1.0*(b/(3.0*a));
                  
                  double largest=2*jj*cos(kk/3.0)-(b/(3.0*a));
                  double middle=L*(M+N)+P;
                  double smallest=L*(M-N)+P;
                  
                  
                  if(largest<middle) //bubble sort; sorting for largest to smallest for eigen values
                      swap(largest,middle);
                  if(middle<smallest)
                      swap(smallest,middle);
                  if(largest<middle)
                      swap(largest,middle);
                  
                  eigVal.at(id).e11=largest;
                  eigVal.at(id).e22=middle;
                  eigVal.at(id).e33=smallest;//eigen values
                  //checking if eigenvalues are nan or not                  
                  if(isnan(largest) || isnan(middle) || isnan(smallest)){
                      std::cout<<"Nan:"<<largest<<"  "<<middle<<"  "<<smallest<<std::endl;
                      std::cout<<M<<"  "<<N<<"  "<<kk<<"  "<<g<<"  "<<ii<<std::endl;
                      std::cout<<i<<"  "<<j<<"  "<<k<<std::endl;
                      std::cout<<"The tau  matrix is..."<<std::endl;
                      display(tau.at(id));
                      std::cout<<"The original matrix is..."<<std::endl;
                      display(mat9);

                  }
              }
	  }// while imaginary
	  /*	  int inn=-82;
	  int jnn=46;
	  int knn=2;*/
	    
	  //eigen Values has to be negative!!!
	  double snumm=0.;
	  double lnumm=100.;

	  //	  if(eigVal.at(id).e11>numm ||eigVal.at(id).e22>numm ||eigVal.at(id).e33>numm || (i==inn && j==jnn && k==knn)){
	  if(eigVal.at(id).e11>=snumm &&eigVal.at(id).e11<lnumm){

              //	    outfile<<i<<"   "<<j<<"   "<<k<<"  "<<std::endl;
	    //std::cout<<i<<"   "<<j<<"   "<<k<<"  "<<" ya ya here"<<std::endl;
	    if(eigVal.at(id).e11<-1.0)std::cout<<"Eigen: "<<eigVal.at(id).e11<<std::endl;
	    // getchar();
	    number++;
	    flagnum=1;
	    
	    
	    /*std::cout<<"Eigen Values are positive"<<std::endl;
	    std::cout<<"The original matrix is..."<<std::endl;
	    display(mat9);
	    
	    std::cout<<"Eigen Values are :"<<std::endl;
	    display(eigVal.at(id));
	    std::cout<<"The d(tau)/dx  matrix is..."<<std::endl;
	    display(taudx.at(id));
	    std::cout<<"Tau matrices FOR cal  d(tau)/dx  matrix are (p and m)..."<<std::endl;
	    display(tau.at(idXp1));
	    display(tau.at(idXm1));
	    std::cout<<"The d(tau)/dy  matrix is..."<<std::endl;
	    display(taudy.at(id));
	    std::cout<<"Tau matrices FOR cal  d(tau)/dy  matrix are (p and m)..."<<std::endl;
	    display(tau.at(idYp1));
	    display(tau.at(idYm1));
	    std::cout<<"The d(tau)/dz  matrix is..."<<std::endl;
	    display(taudz.at(id));
	    std::cout<<"Tau matrices FOR  cal d(tau)/dz  matrix are (p and m)..."<<std::endl;
	    display(tau.at(idZp1));
	    display(tau.at(idZm1));
	    
	    std::cout<<"The tau  matrix is..."<<std::endl;
	    display(tau.at(id));

	    std::cout<<"Realizibility Condition:"<<std::endl;
	    if(tau.at(id).e11<0.0 || tau.at(id).e22<0.0 || tau.at(id).e33<0.0){
	      std::cout<<"One of the following is Negative:"<<std::endl;
	      std::cout<<"tau11="<<tau.at(id).e11<<"/n"
		       <<"tau22="<<tau.at(id).e22<<"/n"
		"tau33="<<tau.at(id).e33<<std::endl;

	    }
	    else{
	      std::cout<<"Diagonal elements are Positive! GOOD!"<<std::endl;
	    }
	    std::cout<<"Checking Diagonal stresses:"<<std::endl;
	    if(fabs(tau.at(id).e12)>sqrt(tau.at(id).e11*tau.at(id).e22)){
	      std::cout<<"fabs(tau12)>sqrt(tau11*tau22)"<<std::endl;
	      std::cout<<"fabs(tau12):"<<fabs(tau.at(id).e12)<<std::endl;
	      std::cout<<"sqrt(tau11*tau22)"<<sqrt(tau.at(id).e11*tau.at(id).e22)<<std::endl;
	    }
	    else{
	      std::cout<<"OffDiagonal Stresses are Okay! (tau12) GOOD!"<<std::endl;
	    }

	    if(fabs(tau.at(id).e13)>sqrt(tau.at(id).e11*tau.at(id).e33)){
	      std::cout<<"fabs(tau13)>sqrt(tau11*tau33)"<<std::endl;
	      std::cout<<"fabs(tau13):"<<fabs(tau.at(id).e13)<<std::endl;
	      std::cout<<"sqrt(tau11*tau33)"<<sqrt(tau.at(id).e11*tau.at(id).e33)<<std::endl;
	    }
	    else{
	      std::cout<<"OffDiagonal Stresses are Okay! (tau13) GOOD!"<<std::endl;
	    }

	    if(fabs(tau.at(id).e23)>sqrt(tau.at(id).e22*tau.at(id).e33)){
	      std::cout<<"fabs(tau23)>sqrt(tau22*tau33)"<<std::endl;
	      std::cout<<"fabs(tau23):"<<fabs(tau.at(id).e23)<<std::endl;
	      std::cout<<"sqrt(tau22*tau33)"<<sqrt(tau.at(id).e22*tau.at(id).e33)<<std::endl;
	    }
	    else{
	      std::cout<<"OffDiagonal Stresses are Okay! (tau23) GOOD!"<<std::endl;
	    }
	    if(matrixDet(tau.at(id))<0.0){
	      std::cout<<"Determinant is Negative! BAD!!"<<std::endl;
	      std::cout<<"Determinant:"<<matrixDet(tau.at(id))<<std::endl;
	    }
	    else{
	      std::cout<<"Determinant is positive! GOOD!"<<std::endl;
	    }
	    
	    
	    
	    std::cout<<"The lamda  matrix is..."<<std::endl;
	    display(lam.at(id));
	    std::cout<<"The CoEps is..."<<std::endl;
	    std::cout<<CoEps.at(id)<<std::endl;
	    
	    std::cout<<"indicies at which this happend are (i,j,k) :"<<i<<"   "<<j<<"   "<<k<<std::endl;
	    getchar();*/
	    // exit(1);
	  }
	  

	  double larMidFac=(eigVal.at(id).e11-eigVal.at(id).e22)/50.0;
	  double firstVal=eigVal.at(id).e11+larMidFac;
	  
	  double smallMidFac=(eigVal.at(id).e22-eigVal.at(id).e33)/50.0;
	  double thirdVal=eigVal.at(id).e33-smallMidFac;
	  
	  double secondVal=0.0;
	  if(larMidFac>smallMidFac)
	    secondVal=eigVal.at(id).e22+larMidFac;
	  else
	    secondVal=eigVal.at(id).e22-smallMidFac;
	  
	  
	  double eigValData[]={firstVal, secondVal, thirdVal};
	  
	  
	  
	  matrix9 eye;//identity matrix
	  
	  eye.e11=1;
	  eye.e12=0;
	  eye.e13=0;
	  eye.e21=0;
	  eye.e22=1;
	  eye.e23=0;
	  eye.e31=0;
	  eye.e32=0;
	  eye.e33=1;
	  
	  for(int ieigen=0;ieigen<3;ieigen++){
	    
	    vec3 vecX;
	    vecX.e11=1.0;
	    vecX.e21=1.0;
	    vecX.e31=1.0;
	    
	    double err=1000;//initial error
	    double s=eigValData[ieigen];
	    
	    while(err>1.0e-5){
	      
	      double maxVec1=maxValAbs(vecX);
	      matrix9 idenEigVal=matrixScalarMult(eye,s);
	      matrix9 matSubs=matrixSubs(mat9,idenEigVal);
	      matrix9 matSubsInv=matrixInv(matSubs);
	      vec3 vecY=matrixVecMult(matSubsInv,vecX);
	      vecX=vecScalarDiv(vecY,vecNorm(vecY));
	      double maxVec2=maxValAbs(vecX);
	      if(maxVec1==0.0){
		std::cerr<<"Divide by Zero!!! (Eulerian.cpp -3)"<<std::endl;
		exit(1);
	      }
	      err=fabs((maxVec1-maxVec2)/maxVec1);
	    }
	    if(ieigen==0){
	      eigVec.at(id).e11=vecX.e11;
	      eigVec.at(id).e21=vecX.e21;
	      eigVec.at(id).e31=vecX.e31;	 
	      vec3 temp;
	      temp.e11=eigVec.at(id).e11;
	      temp.e21=eigVec.at(id).e21;
	      temp.e31=eigVec.at(id).e31;
	      if(vecNorm(temp)<0.9 && vecNorm(temp)>1.1){
		std::cerr<<"Vector is not normalized......exiting....."<<std::endl;
		std::cout<<"Norm is :"<<vecNorm(temp)<<std::endl;
		std::cout<<"indicies are: (i,j,k) :"<<i<<"   "<<j<<"   "<<k<<std::endl; 
		exit(1);
	      }
	    }
	    if(ieigen==1){
	      eigVec.at(id).e12=vecX.e11;
	      eigVec.at(id).e22=vecX.e21;
	      eigVec.at(id).e32=vecX.e31;
	      vec3 temp;
	      temp.e11=eigVec.at(id).e12;
	      temp.e21=eigVec.at(id).e22;
	      temp.e31=eigVec.at(id).e32;
	      if(vecNorm(temp)<0.9 && vecNorm(temp)>1.1){
		std::cerr<<"Vector is not normalized......exiting....."<<std::endl;
		std::cout<<"Norm is :"<<vecNorm(temp)<<std::endl;
		std::cout<<"indicies are: (i,j,k) :"<<i<<"   "<<j<<"   "<<k<<std::endl; 
		exit(1);
	      }
	      
	    }
	    if(ieigen==2){
	      eigVec.at(id).e13=vecX.e11;
	      eigVec.at(id).e23=vecX.e21;
	      eigVec.at(id).e33=vecX.e31;
	      vec3 temp;
	      temp.e11=eigVec.at(id).e13;
	      temp.e21=eigVec.at(id).e23;
	      temp.e31=eigVec.at(id).e33;
	      if(vecNorm(temp)<0.9 && vecNorm(temp)>1.1){
		std::cerr<<"Vector is not vecNormalized......exiting....."<<std::endl;
		std::cout<<"Norm is :"<<vecNorm(temp)<<std::endl;
		std::cout<<"indicies are: (i,j,k) :"<<i<<"   "<<j<<"   "<<k<<std::endl; 
		exit(1);
	      }
	      
	    }
	  }
	  
	  eigVecInv.at(id)=matrixInv(eigVec.at(id));
	  
	  vec3 a0;
	  
	  a0.e11=0.5*(taudx.at(id).e11+taudy.at(id).e12+taudz.at(id).e13);
	  a0.e21=0.5*(taudx.at(id).e12+taudy.at(id).e22+taudz.at(id).e23);
	  a0.e31=0.5*(taudx.at(id).e13+taudy.at(id).e23+taudz.at(id).e33);
	  
	  
	  ka0.at(id)=matrixVecMult(eigVecInv.at(id),a0);
	  
	  
	  g2nd.at(id).e11=0.5*(lam.at(id).e11*taudx.at(id).e11+lam.at(id).e21*taudx.at(id).e12
			       +lam.at(id).e31*taudx.at(id).e13);
	  g2nd.at(id).e21=0.5*(lam.at(id).e12*taudy.at(id).e12+lam.at(id).e22*taudy.at(id).e22
			       +lam.at(id).e32*taudy.at(id).e23);
	  g2nd.at(id).e31=0.5*(lam.at(id).e13*taudz.at(id).e13+lam.at(id).e23*taudz.at(id).e23
			       +lam.at(id).e33*taudz.at(id).e33);
	}
	//	CondNum_A1<<i<<"   "<<j<<"   "<<k<<"  "<<cond_A1<<std::endl;
	//Det_A1<<i<<"   "<<j<<"   "<<k<<"  "<<det_A1<<std::endl;
		
      }
	  if(flagnum==1)std::cout<<"Total:  "<<number<<std::endl;
	  //	  getchar();
  
}

void eulerian::swap(double &a,double &b){
  double temp=a;
  a=b;
  b=temp;
}

double  eulerian::matrixDet(const matrix9& mat){
  
  double detMat=(mat.e11*mat.e22*mat.e33)-
    (mat.e11*mat.e23*mat.e32)-
    (mat.e12*mat.e21*mat.e33)+
    (mat.e12*mat.e23*mat.e31)+
    (mat.e13*mat.e21*mat.e32)-
    (mat.e13*mat.e22*mat.e31);
  
  
  return detMat;
}
double eulerian::matrixDet(const matrix6& matIni){
  
  matrix9 mat;
  mat.e11=matIni.e11;
  mat.e12=matIni.e12;
  mat.e13=matIni.e13;
  mat.e21=matIni.e12;
  mat.e22=matIni.e22;
  mat.e23=matIni.e23;
  mat.e31=matIni.e13;
  mat.e32=matIni.e23;
  mat.e33=matIni.e33;


  double detMat=(mat.e11*mat.e22*mat.e33)-
    (mat.e11*mat.e23*mat.e32)-
    (mat.e12*mat.e21*mat.e33)+
    (mat.e12*mat.e23*mat.e31)+
    (mat.e13*mat.e21*mat.e32)-
    (mat.e13*mat.e22*mat.e31);
  
  return detMat;
}




eulerian::matrix9 eulerian::matrixInv(const matrix9& mat){
  
  matrix9 matInv;
  double detMat=(mat.e11*mat.e22*mat.e33)-
    (mat.e11*mat.e23*mat.e32)-
    (mat.e12*mat.e21*mat.e33)+
    (mat.e12*mat.e23*mat.e31)+
    (mat.e13*mat.e21*mat.e32)-
    (mat.e13*mat.e22*mat.e31);

  if(detMat!=0.0){
    
    matInv.e11=( (mat.e22*mat.e33)-(mat.e23*mat.e32) )/detMat;
    matInv.e12=( (mat.e13*mat.e32)-(mat.e12*mat.e33) )/detMat;
    matInv.e13=( (mat.e12*mat.e23)-(mat.e13*mat.e22) )/detMat;
    matInv.e21=( (mat.e23*mat.e31)-(mat.e21*mat.e33) )/detMat;
    matInv.e22=( (mat.e11*mat.e33)-(mat.e13*mat.e31) )/detMat;
    matInv.e23=( (mat.e13*mat.e21)-(mat.e11*mat.e23) )/detMat;
    matInv.e31=( (mat.e21*mat.e32)-(mat.e22*mat.e31) )/detMat;
    matInv.e32=( (mat.e12*mat.e31)-(mat.e11*mat.e32) )/detMat;
    matInv.e33=( (mat.e11*mat.e22)-(mat.e12*mat.e21) )/detMat;
  }
  else{
    display(mat);
    std::cerr<<"Divide by Zero!!! (Eulerian.cpp - 1,mat9)"<<std::endl;
    exit(1);
    
  }

  return matInv;
}


eulerian::matrix9 eulerian::matrixInv(const matrix6& matIni){
  
  matrix9 matInv,mat;
  mat.e11=matIni.e11;
  mat.e12=matIni.e12;
  mat.e13=matIni.e13;
  mat.e21=matIni.e12;
  mat.e22=matIni.e22;
  mat.e23=matIni.e23;
  mat.e31=matIni.e13;
  mat.e32=matIni.e23;
  mat.e33=matIni.e33;
  

  double detMat=(mat.e11*mat.e22*mat.e33)-
    (mat.e11*mat.e23*mat.e32)-
    (mat.e12*mat.e21*mat.e33)+
    (mat.e12*mat.e23*mat.e31)+
    (mat.e13*mat.e21*mat.e32)-
    (mat.e13*mat.e22*mat.e31);

  if(detMat!=0.0){
    
    matInv.e11=( (mat.e22*mat.e33)-(mat.e23*mat.e32) )/detMat;
    matInv.e12=( (mat.e13*mat.e32)-(mat.e12*mat.e33) )/detMat;
    matInv.e13=( (mat.e12*mat.e23)-(mat.e13*mat.e22) )/detMat;
    matInv.e21=( (mat.e23*mat.e31)-(mat.e21*mat.e33) )/detMat;
    matInv.e22=( (mat.e11*mat.e33)-(mat.e13*mat.e31) )/detMat;
    matInv.e23=( (mat.e13*mat.e21)-(mat.e11*mat.e23) )/detMat;
    matInv.e31=( (mat.e21*mat.e32)-(mat.e22*mat.e31) )/detMat;
    matInv.e32=( (mat.e12*mat.e31)-(mat.e11*mat.e32) )/detMat;
    matInv.e33=( (mat.e11*mat.e22)-(mat.e12*mat.e21) )/detMat;
  }
  else{
      std::cout<<matIni.e23<<"  "<<mat.e23<<std::endl;
    display(mat);
    std::cerr<<"Divide by Zero!!! (Eulerian.cpp - 1,mat6)"<<std::endl;
    exit(1);
    
  }

  return matInv;
}

eulerian::matrix9 eulerian::matrixMult(const matrix9& mat1,const matrix9& mat2){

  matrix9 matMult;
  matMult.e11= mat1.e11*mat2.e11 + mat1.e12*mat2.e21 + mat1.e13*mat2.e31;
  matMult.e12= mat1.e11*mat2.e12 + mat1.e12*mat2.e22 + mat1.e13*mat2.e32;
  matMult.e13= mat1.e11*mat2.e13 + mat1.e12*mat2.e23 + mat1.e13*mat2.e33;
  matMult.e21= mat1.e21*mat2.e11 + mat1.e22*mat2.e21 + mat1.e23*mat2.e31;
  matMult.e22= mat1.e21*mat2.e12 + mat1.e22*mat2.e22 + mat1.e23*mat2.e32;
  matMult.e23= mat1.e21*mat2.e13 + mat1.e22*mat2.e23 + mat1.e23*mat2.e33;
  matMult.e31= mat1.e31*mat2.e11 + mat1.e32*mat2.e21 + mat1.e33*mat2.e31;
  matMult.e32= mat1.e31*mat2.e12 + mat1.e32*mat2.e22 + mat1.e33*mat2.e32;
  matMult.e33= mat1.e31*mat2.e13 + mat1.e32*mat2.e23 + mat1.e33*mat2.e33;

  return matMult;
}

eulerian::matrix9 eulerian::matrixScalarMult(const matrix9& mat ,const double& s){
  matrix9 matRet;
  matRet.e11= mat.e11*s;
  matRet.e12= mat.e12*s;
  matRet.e13= mat.e13*s;
  matRet.e21= mat.e21*s;
  matRet.e22= mat.e22*s;
  matRet.e23= mat.e23*s;
  matRet.e31= mat.e31*s;
  matRet.e32= mat.e32*s;
  matRet.e33= mat.e33*s;

  return matRet;

}

eulerian::matrix9 eulerian::matrixSubs(const matrix9& mat1,const matrix9& mat2){

  matrix9 matSubs;
  matSubs.e11= mat1.e11-mat2.e11;
  matSubs.e12= mat1.e12-mat2.e12;
  matSubs.e13= mat1.e13-mat2.e13;
  matSubs.e21= mat1.e21-mat2.e21;
  matSubs.e22= mat1.e22-mat2.e22;
  matSubs.e23= mat1.e23-mat2.e23;
  matSubs.e31= mat1.e31-mat2.e31;
  matSubs.e32= mat1.e32-mat2.e32;
  matSubs.e33= mat1.e33-mat2.e33;

  return matSubs;
}

eulerian::vec3 eulerian::matrixVecMult(const matrix9& mat,const vec3& vec){

  vec3 matVecMult;
  matVecMult.e11= mat.e11*vec.e11 + mat.e12*vec.e21 + mat.e13*vec.e31;
  matVecMult.e21= mat.e21*vec.e11 + mat.e22*vec.e21 + mat.e23*vec.e31;
  matVecMult.e31= mat.e31*vec.e11 + mat.e32*vec.e21 + mat.e33*vec.e31;


  return matVecMult;
}


void eulerian::display(const matrix9& mat){
  std::cout<<std::endl;
  std::cout<<mat.e11<<"  "<<mat.e12<<"  "<<mat.e13<<"  "<<std::endl;
  std::cout<<mat.e21<<"  "<<mat.e22<<"  "<<mat.e23<<"  "<<std::endl;
  std::cout<<mat.e31<<"  "<<mat.e32<<"  "<<mat.e33<<"  "<<std::endl;
  //  getchar();
}

void eulerian::display(const matrix6& mat){
  std::cout<<std::endl;
  std::cout<<mat.e11<<"  "<<mat.e12<<"  "<<mat.e13<<"  "<<std::endl;
  std::cout<<mat.e12<<"  "<<mat.e22<<"  "<<mat.e23<<"  "<<std::endl;
  std::cout<<mat.e13<<"  "<<mat.e23<<"  "<<mat.e33<<"  "<<std::endl;
  //  getchar();
}

void eulerian::display(const vec3& vec){
  std::cout<<std::endl;
  std::cout<<vec.e11<<std::endl;
  std::cout<<vec.e21<<std::endl;
  std::cout<<vec.e31<<std::endl;
  //getchar();

}

void eulerian::display(const diagonal& mat){

  std::cout<<std::endl;
  std::cout<<mat.e11<<"  "<<"0"<<"  "<<"0"<<"  "<<std::endl;
  std::cout<<"0"<<"  "<<mat.e22<<"  "<<"0"<<"  "<<std::endl;
  std::cout<<"0"<<"  "<<"0"<<"  "<<mat.e33<<"  "<<std::endl;
  //getchar();

}

double eulerian::maxValAbs(const vec3& vec){
  double maxAbs;
  vec3 vecAbs;
  vecAbs.e11=fabs(vec.e11);
  vecAbs.e21=fabs(vec.e21);
  vecAbs.e31=fabs(vec.e31);
  
  if(vecAbs.e11>vecAbs.e21){
    if(vecAbs.e31>vecAbs.e11)
      maxAbs=vecAbs.e31;
    else
      maxAbs=vecAbs.e11;
  }
  else{
    if(vecAbs.e31>vecAbs.e21)
      maxAbs=vecAbs.e31;
    else
      maxAbs=vecAbs.e21;
  }
  return maxAbs;
}

eulerian::vec3 eulerian::vecScalarMult(const vec3& vec, const double& s){
  vec3 vecRet;
  vecRet.e11=vec.e11*s;
  vecRet.e21=vec.e21*s;
  vecRet.e31=vec.e31*s;
  
  return vecRet;
}

eulerian::vec3 eulerian::vecScalarDiv(const vec3& vec, const double& s){
  vec3 vecRet;
  if(s!=0.0){
  vecRet.e11=vec.e11/s;
  vecRet.e21=vec.e21/s;
  vecRet.e31=vec.e31/s;
  }
  else{
    std::cerr<<"Divide by ZERO!!! (Eulerian.cpp - 2)"<<std::endl;
    exit(1);
  }
  return vecRet;
}

double eulerian::vecNorm(const vec3& vec){
  return (sqrt(vec.e11*vec.e11 + vec.e21*vec.e21 + vec.e31*vec.e31));
}

double eulerian::matNormFro(const matrix9& mat){

  matrix9 matTrans,matMult;
  matTrans.e11=mat.e11;
  matTrans.e12=mat.e21;
  matTrans.e13=mat.e31;
  matTrans.e21=mat.e12;
  matTrans.e22=mat.e22;
  matTrans.e23=mat.e32;
  matTrans.e31=mat.e13;
  matTrans.e32=mat.e23;
  matTrans.e33=mat.e33;

  matMult=matrixMult(mat,matTrans);
  double sumDiag=matMult.e11+matMult.e22+matMult.e33;
  return sqrt(sumDiag);
}
double eulerian::matNormFro(const matrix6& mat6){
  

  matrix9 mat,matTrans,matMult;

  mat.e11=mat6.e11;
  mat.e12=mat6.e12;
  mat.e13=mat6.e13;
  mat.e21=mat6.e12;
  mat.e22=mat6.e22;
  mat.e23=mat6.e23;
  mat.e31=mat6.e13;
  mat.e32=mat6.e23;
  mat.e33=mat6.e33;



  matTrans.e11=mat.e11;
  matTrans.e12=mat.e21;
  matTrans.e13=mat.e31;
  matTrans.e21=mat.e12;
  matTrans.e22=mat.e22;
  matTrans.e23=mat.e32;
  matTrans.e31=mat.e13;
  matTrans.e32=mat.e23;
  matTrans.e33=mat.e33;

  matMult=matrixMult(mat,matTrans);
  double sumDiag=matMult.e11+matMult.e22+matMult.e33;
  return sqrt(sumDiag);
}
double eulerian::matCondFro(const matrix9& mat){

  matrix9 matInv=matrixInv(mat);
  double normMat=matNormFro(mat);
  double normMatInv=matNormFro(matInv);
  return  normMat*normMatInv ;
 
}
double eulerian::matCondFro(const matrix6& mat6){

  matrix9 matInv=matrixInv(mat6);
  double normMat=matNormFro(mat6);
  double normMatInv=matNormFro(matInv);
  return  normMat*normMatInv ;
 
}


void eulerian::writeFile(const std::vector<matrix6>& mat,const char* str){
  std::ofstream out;
  out.open(str);
  std::cout<<"Writing file : "<<str<< "  ......"<<std::endl;  
  for(int k=0;k<nz;k++)
    for(int j=0;j<ny;j++)
      for(int i=0;i<nx;i++){
	int id=k*ny*nx + j*nx + i;
	out<<i<<"  "<<j<<"  "<<k<<"  "<<mat.at(id).e11<<"  "<<mat.at(id).e22<<"  "<<mat.at(id).e33<<std::endl;
      }
  std::cout<<"Wrote File : "<<str<<std::endl;
}


void eulerian::writeFile(const std::vector<matrix9>& mat,const char* str ){
  std::ofstream out;
  out.open(str);
  std::cout<<"Writing file : "<<str<< "  ......"<<std::endl;  
  for(int k=0;k<nz;k++)
    for(int j=0;j<ny;j++)
      for(int i=0;i<nx;i++){
	int id=k*ny*nx + j*nx + i;
	out<<i<<"  "<<j<<"  "<<k<<"  "<<mat.at(id).e11<<"  "<<mat.at(id).e22<<"  "<<mat.at(id).e33<<std::endl;
      }
  std::cout<<"Wrote File : "<<str<<std::endl;
}

void eulerian::writeFile(const std::vector<diagonal>& mat,const char* str){
  std::ofstream out;
  out.open(str);
  std::cout<<"Writing file : "<<str<< "  ......"<<std::endl;  
  for(int k=0;k<nz;k++)
    for(int j=0;j<ny;j++)
      for(int i=0;i<nx;i++){
	int id=k*ny*nx + j*nx + i;
	out<<i<<"  "<<j<<"  "<<k<<"  "<<mat.at(id).e11<<"  "<<mat.at(id).e22<<"  "<<mat.at(id).e33<<std::endl;
      }
  std::cout<<"Wrote File : "<<str<<std::endl;
}

void eulerian::writeFile(const std::vector<wind>& wind,const char* str ){
  std::ofstream out;
  out.open(str);
  std::cout<<"Writing file : "<<str<< "  ......"<<std::endl;
  for(int k=0;k<nz;k++)
    for(int j=0;j<ny;j++)
      for(int i=0;i<nx;i++){
	int id=k*ny*nx + j*nx + i;
	out<<i<<"  "<<j<<"  "<<k<<"  "<<wind.at(id).u<<"  "<<wind.at(id).v<<"  "<<wind.at(id).w<<std::endl;
      }
  std::cout<<"Wrote File : "<<str<<std::endl;
  out.close();
}

