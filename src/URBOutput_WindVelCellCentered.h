#pragma once

#include <string>

#include "URBGeneralData.h"
#include "URBOutput_Generic.h"

// Specialized output classes that can take URBGeneratlData or
// URBInputData, etc... and dump out reasonably..
class URBOutput_WindVelCellCentered : public URBOutput_Generic
{
 public:
 URBOutput_WindVelCellCentered()
   : URBOutput_Generic()
    {}
  
  URBOutput_WindVelCellCentered(URBGeneralData*,std::string);
  ~URBOutput_WindVelCellCentered()	       
    {}

  
  bool validateFileOtions();
  void save(URBGeneralData*);
  
  
 private:
  std::vector<float> x_out,y_out,z_out;
  std::vector<int> icellflag_out;
  std::vector<double> u_out,v_out,w_out;
  
};
