#ifndef NONLOCALMIXING_H2
#define NONLOCALMIXINH_H2

#include "Turbulence.h"

class nonLocalMixing2 : public turbulence{
 public:
  void createSigTau(Eulerian*, util&);
 protected:

 private:
  double getMinDistance(int,int,int);
  
};

#endif
