#pragma once

#include <string>

#include "URBGeneralData.h"
#include "URBOutput_Generic.h"

// Specialized output classes that can take URBGeneratlData or
// URBInputData, etc... and dump out reasonably..
class URBOutput_Static : public URBOutput_Generic
{
 public:
 URBOutput_Static()
   : URBOutput_Generic()
    {}
  
  URBOutput_Static(URBGeneralData*,std::string);
  ~URBOutput_Static()	       
    {}

  
  bool validateFileOtions();
  void save(URBGeneralData*);
  
  
 private:
  std::vector<double> x_out,y_out,z_out;
    
};
