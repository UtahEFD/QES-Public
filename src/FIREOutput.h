#pragma once

#include <string>
#include <vector>
#include <algorithm>

#include "URBGeneralData.h"
#include "URBInputData.h"
#include "QESNetCDFOutput.h"
#include "Fire.hpp"

/* Specialized output classes derived from QESNetCDFOutput for 
   cell center data (used primarly for vizualization)
*/
class FIREOutput : public QESNetCDFOutput
{
public:
  FIREOutput()
    : QESNetCDFOutput()
  {}
  FIREOutput(URBGeneralData*,Fire*,std::string);
  ~FIREOutput()	       
  {}
  
  void save(float);
  
private:

  URBGeneralData* ugd_;
  Fire* fire_;
  // all possible output fields need to be add to this list


};
