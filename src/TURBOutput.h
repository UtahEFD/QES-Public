#pragma once

#include <string>

#include "TURBGeneralData.h"
#include "NetCDFOutputGeneric.h"

/* Specialized output classes derived from URBOutput_Generic for 
   cell center data (used primarly for vizualization)
*/
class TURBOutput : public NetCDFOutputGeneric
{
public:
  TURBOutput()
    : NetCDFOutputGeneric()
  {}
  
  TURBOutput(TURBGeneralData*,std::string);
  ~TURBOutput()	       
  {}
  
  void save(float);
  
private:
  
  TURBGeneralData* tgd_;  
  
};
