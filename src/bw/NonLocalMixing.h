#ifndef NONLOCALMIXING_H
#define NONLOCALMIXINH_H

#include "Turbulence.h"

class nonLocalMixing : public turbulence{
 public:
  void createSigTau(Eulerian*, util&);
 protected:

 private:
  double getMinDistance(int,int,int);
  
};

#endif
