#pragma once

#include <string>

#include "URBGeneralData.h"
#include "URBInputData.h"
#include "URBOutput_Generic.h"
/* Specialized output classes derived from URBOutput_Generic for 
   cell center data (used primarly for vizualization)
*/
class URBOutput_VizFields : public URBOutput_Generic
{
 public:
 URBOutput_VizFields()
   : URBOutput_Generic()
    {}
  URBOutput_VizFields(URBGeneralData*,std::string);
  ~URBOutput_VizFields()	       
    {}
  
  bool validateFileOtions();
  void save(URBGeneralData*);
  
 private:
  std::vector<float> x_out,y_out,z_out;
  std::vector<int> icellflag_out;
  std::vector<double> u_out,v_out,w_out;
  
};
