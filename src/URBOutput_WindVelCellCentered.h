#pragma once

#include "UrbOutput_Generic.h"

// Specialized output classes that can take URBGeneratlData or
// URBInputData, etc... and dump out reasonably..
class URBOutput_WindVelCellCentered : public URBOutput_Generic
{
 public:
  
  URBOutput_WindVelCellCentered(URBGeneralData *ugd);
  
  bool validateFileOptions();
  
  void save(URBGeneralData *ugd);
  
  /*
    FM -> maybe u_out, v_out, w_out should be here
     private:
     std::vector<int> icellflag_out;
     std::vector<double> u_out,v_out,w_out;
  */
  
};
