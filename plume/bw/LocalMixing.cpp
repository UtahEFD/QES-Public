#include <iostream>
#include "LocalMixing.h"


void localMixing::createSigTau(eulerian* eul,util& utl){

  double vonKar=utl.vonKar;
  double cPope=0.55;
  double sigUOrg= 2.5;//1.8;
  double sigVOrg=1.6;//2.0;
  double sigWOrg=1.3;//1.3;
  double sigUConst=1.5*sigUOrg*sigUOrg*cPope*cPope;//2.3438;
  double sigVConst=1.5*sigVOrg*sigVOrg*cPope*cPope;//1.5;
  double sigWConst=1.5*sigWOrg*sigWOrg*cPope*cPope;//0.6338;

  std::cout<<"IN Local Mixing Func"<<std::endl;

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
  //  eul->mixLen.resize(nx*ny*nz);

  for(int i=0;i<numBuild;i++){
    xfo.at(i)=utl.xfo.at(i);
    yfo.at(i)=utl.yfo.at(i);
    zfo.at(i)=utl.zfo.at(i);
    hgt.at(i)=utl.hgt.at(i);
    wth.at(i)=utl.wth.at(i);
    len.at(i)=utl.len.at(i);
  }

  int ii=34;
  int jj=7;
  int kk=2;

 ///////////////////doing nothing here//////////////
//   std::ofstream minDist;
//   minDist.open("detMat.dat");
//   if(!minDist.is_open()){
//     std::cerr<<"CANNOT OPEN minDist"<<std::endl;
//     exit(1);
//   }

  double minDistance=0.0;
  double detMat=0.0;
  double condNo=0.0;

  int id;
  for(int k=0;k<nz;k++){
    for(int j=0; j<ny;j++){
      for(int i=0;i<nx;i++){

	double tau11=0.0;
	double tau12=0.0;
	double tau13=0.0;
	double tau22=0.0;
	double tau23=0.0;
	double tau33=0.0;


	id = k*ny*nx + j*nx + i;
	minDistance=0.0;
	detMat=0.0;
	condNo=0.0;
	if(eul->CellType.at(id).c!=0){ //Calculate gradients ONLY if it is a fluid cell
	  minDistance = getMinDistance(i ,j ,k );

	  double Lm  = 0.0;
	  double S11 = 0.0;
	  double S22 = 0.0;
	  double S33 = 0.0;
	  double S12 = 0.0;
	  double S23 = 0.0;
	  double S13 = 0.0;
	  double nu_T = 0.0;
	  double Tke = 0.0;
          //          eul->mixLen.at(id)=Lm;


          if(i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1){


              int idXm1= k*ny*nx + j*nx + (i-1);
              int idXp1= k*ny*nx + j*nx + (i+1);

              int idYm1= k*ny*nx + (j-1)*nx + i;
              int idYp1= k*ny*nx + (j+1)*nx + i;

              int idZm1= (k-1)*ny*nx + j*nx + i;
              int idZp1= (k+1)*ny*nx + j*nx + i;

              // For wind gradients in X-direction

              double dUdx=0.0;
              double dVdx=0.0;
              double dWdx=0.0;

              // Three cases- 1) cells are at boundaries,2) cells are right next to the wall, 3) all other cases (fluid cells)

              if(i==0 || i==nx){//boundaries
                  dUdx = 0.0;
                  dVdx = 0.0;
                  dWdx = 0.0;
              }else if(eul->CellType.at(idXm1).c==0 || eul->CellType.at(idXp1).c==0){
                  dUdx = eul->windVec.at(id).u/(0.5*log(0.5/utl.zo));
                  dVdx = eul->windVec.at(id).v/(0.5*log(0.5/utl.zo));
                  dWdx = eul->windVec.at(id).w/(0.5*log(0.5/utl.zo));
              }else{
                  dUdx = (eul->windVec.at(idXp1).u - eul->windVec.at(idXm1).u)/(2.0);
                  dVdx = (eul->windVec.at(idXp1).v - eul->windVec.at(idXm1).v)/(2.0);
                  dWdx = (eul->windVec.at(idXp1).w - eul->windVec.at(idXm1).w)/(2.0);
              }
              // For wind gradients in Y-direction

              double dUdy=0.0;
              double dVdy=0.0;
              double dWdy=0.0;

              // Three cases- 1) cells are at boundaries,2) cells are right next to the wall, 3) all other cases (fluid cells)

              if(j==0 || j==ny){//boundaries
                  dUdy = 0.0;
                  dVdy = 0.0;
                  dWdy = 0.0;
              }else if(eul->CellType.at(idYm1).c==0 || eul->CellType.at(idYp1).c==0){
                  dUdy = eul->windVec.at(id).u/(0.5*log(0.5/utl.zo));
                  dVdy = eul->windVec.at(id).v/(0.5*log(0.5/utl.zo));
                  dWdy = eul->windVec.at(id).w/(0.5*log(0.5/utl.zo));
              }else{
                  dUdy = (eul->windVec.at(idYp1).u - eul->windVec.at(idYm1).u)/(2.0);
                  dVdy = (eul->windVec.at(idYp1).v - eul->windVec.at(idYm1).v)/(2.0);
                  dWdy = (eul->windVec.at(idYp1).w - eul->windVec.at(idYm1).w)/(2.0);
              }

              // For wind gradients in Z-direction

              double dUdz=0.0;
              double dVdz=0.0;
              double dWdz=0.0;

              //Three cases-1) cells just above ground or rooftop,2)cells at the top boundary, 3)all other cases (fluid cells)

              if(k==0){//ground
                  dUdz=0.0;
                  dVdz=0.0;
                  dWdz=0.0;

              }
              else if(k==1 || (eul->CellType.at(idXm1).c==0 && eul->CellType.at(id).c==1)){//just above ground, log law
                  dUdz = eul->windVec.at(id).u / (minDistance*(log(minDistance/utl.zo)));
                  dVdz = eul->windVec.at(id).v / (minDistance*(log(minDistance/utl.zo)));
                  dWdz = eul->windVec.at(id).w / (minDistance*(log(minDistance/utl.zo)));
              }
              else if(k==nz-1){
                  dUdz=0.0;//dudz.at(idZm1);
                  dVdz=0.0;//dudz.at(idZm1);
                  dWdz=0.0;//dudz.at(idZm1);
              }
              else{
                  dUdz= ( eul->windVec.at(idZp1).u-eul->windVec.at(idZm1).u ) / (2.0);
                  dVdz= ( eul->windVec.at(idZp1).v-eul->windVec.at(idZm1).v ) / (2.0);
                  dWdz= ( eul->windVec.at(idZp1).w-eul->windVec.at(idZm1).w ) / (2.0);
              }


              Lm = vonKar*minDistance;
              //              eul->mixLen.at(id)=Lm;
              S11 = dUdx;
              S22 = dVdy;
              S33 = dWdz;
              S12 = 0.5*(dUdy+dVdx);
              S23 = 0.5*(dVdz+dWdy);
              S13 = 0.5*(dUdz+dWdx);

              double SijSij=S11*S11 + S22*S22 + S33*S33 + 2.0*(S12*S12 + S13*S13 + S23*S23);

              nu_T = Lm*Lm * sqrt(2.0*SijSij);

              Tke = pow( (nu_T/(cPope*Lm)) ,2.0);

              tau11=(2.0/3.0) * Tke - 2.0*(nu_T*S11);
              tau22=(2.0/3.0) * Tke - 2.0*(nu_T*S22);
              tau33=(2.0/3.0) * Tke - 2.0*(nu_T*S33);
              tau12= - 2.0*(nu_T*S12);
              tau13= - 2.0*(nu_T*S13);
              tau23= - 2.0*(nu_T*S23);


              /*if(fabs(tau11)<(fabs(tau12)+fabs(tau13))){
                tau11=(fabs(tau12)+fabs(tau13));
                //std::cout<<"adjusted at tau11 :" <<i<<"   "<<j<<"   "<<k<<std::endl;
                }

                if(fabs(tau22)<(fabs(tau12)+fabs(tau23))){
                tau22=(fabs(tau12)+fabs(tau23));
                //std::cout<<"adjusted at tau22 :" <<i<<"   "<<j<<"   "<<k<<std::endl;
                }

                if(fabs(tau33)<(fabs(tau13)+fabs(tau23))){
                tau33=(fabs(tau13)+fabs(tau23));
                //std::cout<<"adjusted tau33 at :" <<i<<"   "<<j<<"   "<<k<<std::endl;
                }*/

              tau11=fabs(sigUConst*tau11);
              tau22=fabs(sigVConst*tau22);
              tau33=fabs(sigWConst*tau33);
              
              //add non-local
              if(k<9){
                  double constant=.3;
                  tau11=tau11+constant;
                  tau22=tau22+constant;
                  tau33=tau33+constant;
                  tau12=-tau12;
                  tau13=-tau13;
                  tau23=-tau23;
              }
              else{
                  double constant=.3;
                  tau11=tau11+constant;
                  tau22=tau22+constant;
                  tau33=tau33+constant;
                  tau12=-tau12;
                  tau13=-tau13;
                  tau23=-tau23;
              }
              
              //end non-local

              eul->tau.at(id).e11 = tau11;//adjust vertical gradients here
              eul->tau.at(id).e22 = tau22;
              eul->tau.at(id).e33 = tau33;
              eul->tau.at(id).e12 = tau12;
              eul->tau.at(id).e13 = tau13;
              eul->tau.at(id).e23 = tau23;
              eul->lam.at(id)=eul->matrixInv(eul->tau.at(id));

              //	    detMat=eul->matrixDet(eul->tau.at(id));
              //condNo=eul->matCondFro(eul->tau.at(id));
              /*if(k>0 && detMat>1.0){
                eul->display(eul->tau.at(id));
                std::cout<<detMat<<"  "<<condNo<< std::endl;
                getchar();
                }*/




              eul->sig.at(id).e11 = pow(eul->tau.at(id).e11,0.5);
              eul->sig.at(id).e22 = pow(eul->tau.at(id).e22,0.5);
              eul->sig.at(id).e33 = pow(eul->tau.at(id).e33,0.5);
              double ustarCoEps=sqrt(Tke)*cPope;
              //            std::cout<<"ddddd"<<std::endl;
              eul->CoEps.at(id)=5.7* pow(ustarCoEps,3.0)/(Lm);

              if(i==72 && (j==49 || j==60) && k==1){
                  /*std::cout<<std::endl;
                    std::cout<<"J : "<<j<<std::endl;
                    std::cout<<eul->windVec.at(id).u<<"   "<<eul->windVec.at(id).v<<"  "<<eul->windVec.at(id).w<<std::endl;
                    eul->display(eul->tau.at(id));
                    eul->display(eul->lam.at(id));*/

              }
     	  }// if for domain
        }//if for celltype
      }
    }
  }
  std::cout<<"num of LocalMixing data is:"<<id<<"\n";

}

double localMixing::getMinDistance(int i, int j, int qk){

  std::vector<double> distance; //stores minimum distance of (i,j,k) to each building

  // Adding 0.5*gridResolution as the QUIC-URB grid is shifted by 0.5*gridResolution
  double iCell=i+0.5*dx; // converting units in meters (original position of the cell in meters)
  double jCell=j+0.5*dy;
  double kCell=qk+0.5*dz-1.0;//TEMPORARY SUBTRACT 1---******************************CHECK THIS*****

  for(int build=0;build<numBuild;++build){

    double x=xfo.at(build); //storing the building parameters
    double y=yfo.at(build);
    double l=len.at(build);
    double w=wth.at(build);
    double h=hgt.at(build);

    double minDisFaces=0.0; //minimum distance to 4 faces(sides) from a cell
    double minDisTop=0.0;//minimum distance to the top(roof) of the building from a cell

    if(kCell<h){//For this condition we have only 4 planes for each building

      double actualDis[4];// absolute value of the perpendDis or the actual distance, we have 4 faces

      for(int i=0;i<4;++i){
	// i=0 is front face
	// i=1, back face
	// i=2, right side face (facing towards front face of the building)
	// i=3, left side face:

	double iedge;  // edges of the suface, declared as doubles as ...
	double jedge;  // one of the edge value for cells perpendicular...


	if(i==0 ){//front face
	  int edge1=(int)(y-(w/2));//right edge of the front plane
	  int edge2=(int)(y+(w/2));//left edge of the front plane
	  jedge=x;// to get the edge in X-Direction
	  if( jCell<=edge1 || jCell>=edge2 ){//for cells (i,j,qk) off the plane
	    if(fabs(edge2-jCell)< fabs(edge1-jCell))//for cells which are closer to "edge2"
	      iedge=edge2;
	    else
	      iedge=edge1;

	  }
	  else{ //for cells perpendicular to the faces
	    iedge=jCell;
	  }
	  actualDis[i]=pow( (pow((iCell-jedge),2.0f)) + (pow((jCell-iedge),2.0f)) , 0.5f );


	}// if condition for i==0 ends

	if(i==1){//back face
	  int edge1=(int)(y-(w/2));
	  int edge2=(int)(y+(w/2));
	  jedge=x+l; //back face
	  if(jCell<edge1 || jCell>edge2){
	    if(fabs(edge2-jCell)< fabs(edge1-jCell)) //for cells which are closer to "edge2"
	      iedge=edge2;
	    else
	      iedge=edge1;
	  }
	  else{
	    iedge=jCell;
	  }
	  actualDis[i]=pow( (pow((iCell-jedge),2.0f)) + (pow((jCell-iedge),2.0f)) , 0.5f );
	}//if condition for i==1 ends

	if(i==2){//right side face
	  int edge1=(int)(x);
	  int edge2=(int)(x+l);
	  iedge=y-(w/2);
	  if(iCell>edge2 || iCell<edge1){
	    if(fabs(edge1-iCell) < fabs(edge2-iCell))
	      jedge=edge1;
	    else
	      jedge=edge2;
	  }
	  else{
	    jedge=iCell;
	  }
	  actualDis[i]=pow( (pow((iCell-jedge),2.0f)) + (pow((jCell-iedge),2.0f)) , 0.5f );
	}//if condition for i==2 ends
	if(i==3){// left side face
	  int edge1=(int)(x);
	  int edge2=(int)(x+l);
	  iedge=y+(w/2);
	  if(iCell>edge2 || iCell<edge1){
	    if(fabs(edge1-iCell) < fabs(edge2-iCell))
	      jedge=edge1;
	    else
	      jedge=edge2;
	  }
	  else{
	    jedge=iCell;
	  }
	  actualDis[i]=pow( (pow((iCell-jedge),2.0f)) + (pow((jCell-iedge),2.0f)) , 0.5f );
	}//if condition for i==3 ends
      }// For Loop for number of faces ends
      minDisFaces=actualDis[1];//assuming one is minimum

      for(int i=0;i<(int)(sizeof(actualDis)/sizeof(*actualDis));++i){  //sizeof() provide number of bytes
	if(minDisFaces>actualDis[i])
	  minDisFaces=actualDis[i];
// 	  std::cout<<i<<"\n";
      }

      if(minDisFaces>kCell) // checking if ground is closer than any of the faces
        minDisFaces=kCell;

	  distance.push_back(minDisFaces);
    }
    else{ //if qk>=h

//   std::cout<<" qk>=h"<<"\n";
      double iedge;
      double jedge = 0.0;
      double kedge;

      int edgeX1=(int)(x);
      int edgeX2=(int)(x+l);
      int edgeY1=(int)(y-(w/2));
      int edgeY2=(int)(y+(w/2));

      if((iCell<edgeX1 || iCell>edgeX2 || jCell<edgeY1 || jCell>edgeY2)  ) { // for all the off plane cells (areas B0 and B1 in the PPT)
        iedge=jCell;
	kedge=h;
	if(iCell<=edgeX1){ // cells in front of front face
	  jedge=edgeX1;
	  if(jCell<edgeY1)
	    iedge=edgeY1;

	  if(jCell>edgeY2)
	    iedge=edgeY2;
        }
        if(iCell>=edgeX2){//cells behind the back face
          jedge=edgeX2;
	  if(jCell<=edgeY1)
	    iedge=edgeY1;
	  if(jCell>edgeY2)
	    iedge=edgeY2;
	}
	if(iCell>edgeX1 && iCell<edgeX2){ //cells  on either side of side faces

	  jedge=iCell;
	  kedge=h;
	  if(jCell<=edgeY1)
	    iedge=edgeY1;
	  if(jCell>edgeY2)
	    iedge=edgeY2;
	}

      }
      else{//if the prependicular from the cell lies on the roof.
	iedge=jCell;
	jedge=iCell;
	kedge=h;
      }

      minDisTop=pow( (pow((iCell-jedge),2.0f)) + (pow((jCell-iedge),2.0f)) + (pow((kCell-kedge),2.0f))  , 0.5f );
      if(minDisTop>kCell) // checking if ground is closer than the distance to the roof.
	    minDisTop=kCell;

      distance.push_back(minDisTop);

    }//if else of qk>h or qk<h ends
  }//For loop for buildings

  std::sort(distance.begin(),distance.end());

  return distance[0];// returning smallest distance

  }































	/*
	//double VertGradFactor=pow( (1.0-(minDistance/20.0)) ,3.0/4.0);
	ustar=0.4*minDistance*du_dz; //Note: ustar doesn't include the vertGradFactor; sigmas do have vertGradFactor

	data3[texidx] = du_dz; //du_dz
	data3[texidx+1] = 0.0; //dv_dz
	data3[texidx+2] = VertGradFactor;    //dw_dz
	  data3[texidx+3] = ustar;

	  sigU = 2.5*ustar*VertGradFactor;
	  sigV = 2.0*ustar*VertGradFactor;
	  sigW = 1.3*ustar*VertGradFactor;

	  sig[p2idx].u = sigU;   //sigU
	  sig[p2idx].v = sigV;   //sigV
	  sig[p2idx].w = sigW;   //sigW

	  tau11=sigU*sigU;
	  tau22=sigV*sigV;
	  tau33=sigW*sigW;
	  tau13=ustar*ustar;
	  double tauDetInv=1.0f/((tau11*tau22*tau33)-(tau13*tau13*tau22));

	  updateMaxandMinTaus(tau11,tau22,tau33,tau13);

	  tau[p2idx].t11   = tau11;             //Tau11
	  tau[p2idx+1].t22 = tau22;             //Tau22
	  tau[p2idx+2].t33 = tau33;             //Tau33
	  tau[p2idx+3].t13 = tau13;             //Tau13
	  //Make tau's a texture so that they can be visualized as horizontal layers in the domain
	  dataTau[texidx] = tau11;
	  dataTau[texidx+1] = tau22;
	  dataTau[texidx+2] = tau33;
	  dataTau[texidx+3] = tau13;

	  dataTwo[texidx]   =  1.0f/(tau11-tau13*tau13/tau33);// tauDetInv*(tau22*tau33);   //Lam11
	  dataTwo[texidx+1] =  1.0f/tau22;// tauDetInv*(tau11*tau33-tau13*tau13);           //Lam22
	  dataTwo[texidx+2] =  1.0f/(tau33-tau13*tau13/tau11);//tauDetInv*(tau11*tau22);    //Lam33
	  dataTwo[texidx+3] =  -tau13/(tau11*tau33-tau13*tau13);//tauDetInv*(-tau13*tau22); //Lam13



   */
