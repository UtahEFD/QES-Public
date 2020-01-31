#pragma once

#include <cmath>
#include "util/ParseInterface.h"
#include "PolyBuilding.h"

enum CanopyType {
  Cionco,
  Vineyard
};

class Canopy : public PolyBuilding
{
public:
  
  Canopy()
  { 
  }
  virtual ~Canopy()
  {
  }
  
  CanopyType _cType;

protected: 
  /*!
   * This function takes in variables initialized by the readCanopy function and sets the boundaries of 
   * the canopy and defines initial values for the canopy height.
   */
  void canopyDefineBoundary(URBGeneralData *ugd,int cellFlagToUse);

  /*!
   * For there and below, the canopyVegetation function has to be defined
   */
  virtual void canopyVegetation(URBGeneralData *ugd) = 0;
  
private:
  
};
