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
  
};
