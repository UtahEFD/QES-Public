#ifndef DISPERSION_H
#define DISPERSION_H

#include <list>
#include <vector>
#include <iostream>
#include "Eulerian.h"
#include "Random.h"
#include <cutil_math.h>
class dispersion{
 public:
  eulerian eul; 
  void createDisp(const eulerian&);
  struct matrix{
    double x;
    double y;
    double z;
  };
  
 // std::vector<matrix> pos,prime;
  std::vector<float3> pos,prime;
  std::list<double> zIniPos;
  std::list<double> wPrime;
  double eps;
  int numTimeStep;
  std::vector<double> timeStepStamp,tStrt;
  int parPerTimestep;
  
 private:
  
  double xSrc,ySrc,zSrc;
  int numPar;
	
};
#endif
