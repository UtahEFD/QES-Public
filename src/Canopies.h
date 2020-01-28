#pragma once

#include "util/ParseInterface.h"
#include "Building.h"
#include "Canopy.h"
#include "URBInputData.h"
#include "URBGeneralData.h"

class Canopies : public ParseInterface
{
 private:
  
  
  
 public:
  
  int num_canopies;
  std::vector<Building*> canopies;
  

  virtual void parseValues()
  {
    parsePrimitive<int>(true, num_canopies, "num_canopies");
    parseMultiPolymorphs(true, canopies, Polymorph<Building, Canopy>("canopy"));
    
  }
};
