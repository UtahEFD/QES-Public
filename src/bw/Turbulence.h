#ifndef Turbulence_H
#define Turbulence_H
#include <math.h>
#include <vector>
#include "Eulerian.h"
#include "Util.h"


class turbulence{
 public:
  virtual void createSigTau(Eulerian*, util&); 
  std::vector<double> xfo,yfo,zfo,hgt,wth,len;
  int numBuild,nx,ny,nz;
  double dx,dy,dz;
 protected:

 private:
  
};
#endif
