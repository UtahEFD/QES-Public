#pragma once

#include <string>

#include "URBGeneralData.h"
#include "NetCDFOutputGeneric.h"

/* Specialized output classes derived from NetCDFOutputGeneric for 
   face center data (used for turbulence,...)
*/
class URBOutputWorkspace : public NetCDFOutputGeneric
{
public:
  URBOutputWorkspace()
    : NetCDFOutputGeneric()
  {}
  
  URBOutputWorkspace(URBGeneralData*,std::string);
  ~URBOutputWorkspace()	       
  {}

  void save(float);
 
private:
  std::vector<float> x_cc,y_cc,z_cc;

  URBGeneralData* ugd_;

};
