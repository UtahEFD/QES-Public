#ifndef LOCALMIXING_H
#define LOCALMIXINH_H

#include "Turbulence.h"

class localMixing : public turbulence{
 public:
  void createSigTau(eulerian*, util&);
 protected:

 private:
  double getMinDistance(int,int,int);
  
};

#endif
