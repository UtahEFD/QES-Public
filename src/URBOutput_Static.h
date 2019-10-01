#pragma once

#include <string>

#include "URBGeneralData.h"
#include "URBOutput_Generic.h"

/* Specialized output classes derived from URBOutput_Generic for 
   static data (terrain,...)
   Note: Need to implement building here!!
*/

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
  std::vector<float> x_out,y_out,z_out;
  
};
