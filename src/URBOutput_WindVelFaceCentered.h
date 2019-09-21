#pragma once

#include <string>

#include "URBGeneralData.h"
#include "URBOutput_Generic.h"

// Specialized output classes that can take URBGeneratlData or
// URBInputData, etc... and dump out reasonably..
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
  // - no copy are needed here
  // std::vector<double> x_out,y_out,z_out;
  // std::vector<double> u_out,v_out,w_out;
  
};
