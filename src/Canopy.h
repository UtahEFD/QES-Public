#pragma once

#include <cmath>
#include "util/ParseInterface.h"
#include "PolyBuilding.h"

class Canopy : public PolyBuilding
{
public:
  
  Canopy()
  { 
  }
  virtual ~Canopy()
  {
  }
  
protected: 
  /*!
   * This function takes in variables initialized by the readCanopy function and sets the boundaries of 
   * the canopy and defines initial values for the canopy height.
   */
  void canopyDefineBoundary(URBGeneralData *ugd,int cellFlagToUse);
  
private:
  
};
