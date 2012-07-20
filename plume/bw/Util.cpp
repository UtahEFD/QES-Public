#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

#include "Util.h"
util::util(): twidth(0),numPar(0), nx(0),ny(0),nz(0), theight(0), windFieldData(0),
	      numBoxX(0),numBoxY(0),numBoxZ(0),timeStep(0.0),ustar(0.0),dur(0.0),sCBoxTime(0.0),eCBoxTime(0.0), 
	      avgTime(0.0),xSrc(0.0),ySrc(0.0),zSrc(0.0),rSrc(0.0),
	      xBoxSize(0.0),yBoxSize(0.0),zBoxSize(0.0),zo(0.0),
	      profile(0)
{
  std::fill(bnds,bnds+6,0.0);
}



void util::readInputFile(){
  zo=0.006667;

  dx=1.0;
  dy=1.0;
  dz=1.0;
  vonKar=0.4;

  ibuild=0;

  std::ifstream in;
  in.open("../bw/input.txt");
  
  char line[1024];
  std::string inputStr;
  
  while(!in.eof()){
    in.getline(line,1024);
    
    if(line[0]!='#' && strlen(line)!=0){
      std::istringstream str(line);
      str >> inputStr;
      
      if(inputStr=="twidth"){
	str >> twidth;//1000
      }
      if(inputStr=="theight"){
	str >> theight;//100
	numPar=twidth*theight;
      }
      if(inputStr=="nx"){
	str >> nx;//153
      }
      if(inputStr=="ny"){
	str >> ny;//110
      }
      if(inputStr=="nz"){
	str >> nz;//30
      }
      if(inputStr=="windFieldData"){
	str >> windFieldData;
      }
      if(inputStr=="time_step"){
	str >> timeStep;
      }
      if(inputStr=="output_file"){
	str >> file;
      }
      if(inputStr=="duration"){
	str >> dur;
      }
      if(inputStr=="ustar"){
	str >> ustar;
      }
      if(inputStr=="startCBoxTime"){
	str >> sCBoxTime;
      }
      if(inputStr=="endCBoxTime"){
	str >> eCBoxTime;
      }
      if(inputStr=="averagingTime"){
	str >> avgTime;
      }
      if(inputStr=="bounds"){
	for(int i=0;i<6;i++)
	  str>>bnds[i];
      }
      if(inputStr=="numBox_x"){
	str >> numBoxX;
	xBoxSize=(bnds[1]-bnds[0])/numBoxX;
      }
      if(inputStr=="numBox_y"){
	str >> numBoxY;
	yBoxSize=(bnds[3]-bnds[2])/numBoxY;
      }
      if(inputStr=="numBox_z"){
	str >> numBoxZ;
	zBoxSize=(bnds[5]-bnds[4])/numBoxZ;
      }
      if(inputStr=="source_info"){
	str >> src;
	if(strcmp(src.c_str(),"point")==0){
	  
	  str >> xSrc;
	  str >> ySrc;
	  str >> zSrc;
	  str >> rSrc;
	}
      }
      if(inputStr=="numBuild"){
	str >> numBuild;
	xfo.resize(numBuild);
	yfo.resize(numBuild);
	zfo.resize(numBuild);
	hgt.resize(numBuild);
	wth.resize(numBuild);
	len.resize(numBuild);
      }
      if(inputStr=="build"){      
	
	str >> xfo.at(ibuild);
	str >> yfo.at(ibuild);
	str >> zfo.at(ibuild);
	str >> hgt.at(ibuild);
	str >> wth.at(ibuild);
	str >> len.at(ibuild);
	ibuild++;
      }
      inputStr="";
    }
  }
  
  /*  twidth=1000;
      theight=10;
      numPar=twidth*theight;
      nx=100;
      ny=50;
      nz=20;
      windFieldData=4;
      timeStep=0.01;
      file="data.txt";
      dur=100;
      ustar=0.084;
      sCBoxTime=10;
      eCBoxTime=10000;
      avgTime=90;
      
      bnds[0]=0;
      bnds[1]=100.0;
      bnds[2]=0;
      bnds[3]=50.0;
      bnds[4]=0;
      bnds[5]=20.0;
      
      numBox=20;
      zBoxSize=(bnds[1]-bnds[0])/numBox;
      zSrc=0;
      rSrc=0;*/
}
