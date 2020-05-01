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

    // data container for output (on cell-center without ghost cell)
    std::vector<float> z_out;
    std::vector<int> icellflag_out;
    std::vector<float> u_out,v_out,w_out;
    
    // copy of pointer for data access
    URBGeneralData* ugd_;
    Fire* fire_;
    
};
