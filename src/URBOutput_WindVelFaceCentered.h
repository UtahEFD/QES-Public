#pragma once

#include <string>

#include "URBGeneralData.h"
#include "URBOutput_Generic.h"

/* Specialized output classes derived from URBOutput_Generic for 
   face center data (used for turbulence,...)
*/
class URBOutput_WindVelFaceCentered : public URBOutput_Generic
{
 public:
 URBOutput_WindVelFaceCentered()
   : URBOutput_Generic()
    {}
  
  URBOutput_WindVelFaceCentered(URBGeneralData*,std::string);
  ~URBOutput_WindVelFaceCentered()	       
    {}

  
  bool validateFileOtions();
  void save(URBGeneralData*);
 
 private:
  
};
