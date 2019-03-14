// 1DTestLangevin.cpp : Defines the entry point for the console application.

#include <iostream>
#include <cstring>
#include "Dispersion.h"
void advectPar(const util&,dispersion&,eulerian&, const char*, const int);
void advectParOLD(const util&,dispersion&,eulerian&, const char*);

int main(int argc, char *argv[]){
  
  /*  if(argc<2){
    std::cout<<std::endl;
    std::cerr<<"ERROR 1: Method Missing"<<std::endl;
    return 1;
    }*/
  if(argc>2){
    std::cout<<std::endl;
    std::cerr<<"ERROR 2: Not an appropriate Method, use - old or new"<<std::endl;
    return 2;
    
  }
  if(argc==2 && std::strcmp(argv[1],"o")!=0 && std::strcmp(argv[1],"n")!=0){
    std::cout<<std::endl;
    std::cerr<<"ERROR 3: Not an appropriate Method, use - old or new"<<std::endl;
    return 3;
    
  }
  
  std::cout<<std::endl;
  std::cout<<"Going to UTL"<<std::endl;
  
  util utl;
  utl.readInputFile("");
  std::cout<<"Going to EUL"<<std::endl;
  
  
  eulerian eul;
  eul.createEul(utl);
  std::cout<<"Going to Disp"<<std::endl;
  
  dispersion disp;
  disp.createDisp(eul);  
  
  advectPar(utl,disp,eul,argv[1],argc);
    //advectParOLD(utl,disp,eul,argv[1]);
  
  return 0;
}

