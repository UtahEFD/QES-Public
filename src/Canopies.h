#pragma once

#include "util/ParseInterface.h"
#include "Building.h"
#include "Canopy.h"
#include "CanopyCionco.h"
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
    // read the input data for canopies
    parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyCionco>("CanopyCionco"));
    // add other type of canopy here
  }
};
