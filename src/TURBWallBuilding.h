#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <limits>

#include "URBGeneralData.h"
#include "TURBGeneralData.h"
#include "TURBWall.h"



class TURBWallBuilding : public TURBWall
{
protected:
public:
  
  TURBWallBuilding()
  {}
  ~TURBWallBuilding()
  {}
  
  /**
   * @brief
   *
   * This function takes in the icellflags set by setCellsFlag
   * function for stair-step method and sets related coefficients to
   * zero to define solid walls. It also creates vectors of indices
   * of the cells that have wall to right/left, wall above/bellow
   * and wall in front/back
   *
   */
  void defineWalls(URBGeneralData*,TURBGeneralData*);
  void setWallsBC(URBGeneralData*,TURBGeneralData*);
  
private:
  const int icellflag_building = 0;
  const int icellflag_cutcell = 7;
  
  const int iturbflag_stairstep = 4;
  const int iturbflag_cutcell = 5;
  
  bool use_cutcell = false;

};
