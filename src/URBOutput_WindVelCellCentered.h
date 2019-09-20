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
  
  /*
    FM -> maybe u_out, v_out, w_out should be here
    private:
    std::vector<int> icellflag_out;
    std::vector<double> u_out,v_out,w_out;
  */
  
};
