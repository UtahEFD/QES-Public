#pragma once

#include <string>

#include "URBGeneralData.h"
#include "URBOutput_Generic.h"

/* Specialized output classes derived from URBOutput_Generic for 
   face center data (used for turbulence,...)
*/
class URBOutput_TURBInputFile : public URBOutput_Generic
{
 public:
 URBOutput_TURBInputFile()
   : URBOutput_Generic()
    {}
  
  URBOutput_TURBInputFile(URBGeneralData*,std::string);
  ~URBOutput_TURBInputFile()	       
    {}

  void save(URBGeneralData*);
 
 private:
    std::vector<float> x_cc,y_cc,z_cc;
};
