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

// may need to forward reference this???
class TURBGeneralData;

class TURBWall
{
public:
  
  TURBWall()
  {}
  ~TURBWall()
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
  virtual void defineWalls(URBGeneralData*,TURBGeneralData*) = 0;
  virtual void setWallsBC(URBGeneralData*,TURBGeneralData*) = 0;
  
protected:

  void get_stairstep_wall_id(URBGeneralData*,int);
  void set_stairstep_wall_flag(TURBGeneralData*,int);
  
  void get_cutcell_wall_id(URBGeneralData*,int);
  void set_cutcell_wall_flag(TURBGeneralData*,int);
  
  void set_loglaw_stairstep_at_id_cc(URBGeneralData*,TURBGeneralData*,int,int,float);
  
  // cells above wall (for stair-step methods for Wall BC)
  std::vector<int> stairstep_wall_id;
  // cut-cell cells
  std::vector<int> cutcell_wall_id;
  
private:
 
};
