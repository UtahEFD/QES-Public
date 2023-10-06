// Smoke.h
// This class specifies the smoke sources for QES-Fire
// Matthew Moody 10/04/2023
//

#ifndef SMOKE_H
#define SMOKE_H

#include "Fire.h"
#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"
#include "util/Vector3.h"
#include "util/Vector3Int.h"
#include "winds/DTEHeightField.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include "SourceFire.h"
#include "plume/Plume.hpp"

using namespace std;
class Fire;
class Plume;
class Smoke
{
public:
  Smoke();
  void genSmoke(WINDSGeneralData *, Fire *, Plume *);

  
  void source();

 private:
  int nx,ny,nz;
  float dx,dy,dz;
  float x_pos,y_pos,z_pos;
  float ppt;
};

#endif

