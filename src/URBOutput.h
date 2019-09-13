#pragma once

#include <vector>
#include <netcdf>
#include <cmath>

#include "URBGeneralData.h"
#include "URBInputData.h"
#include "Output.hpp"


/* output: 
   cell-centered wind velocity for viz
   face-centered wind velocity for TURB
   bulding/terrain ??
    - cut-cell
    - bulding ID (?)
*/


class URBOutput {
 public:
  
  // constructor
  URBOutput(const URBInputData* UID, const URBGeneralData* UGD);
  
  //Save 
  void save(const URBInputData* UID, const URBGeneralData* UGD);
  
  std::vector<Output *> outputs;

};
