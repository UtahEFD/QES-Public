#pragma once

#include <string>

#include "TURBGeneralData.h"
#include "QESNetCDFOutput.h"

/* Specialized output classes derived from QESNetCDFOutput for 
   cell center data (used primarly for vizualization)
*/
class TURBOutput : public QESNetCDFOutput
{
public:
    TURBOutput()
        : QESNetCDFOutput()
    {}
    
    TURBOutput(TURBGeneralData*,std::string);
    ~TURBOutput()	       
    {}
    
    void save(float);
    
private:
    
    TURBGeneralData* tgd_;  
    
};
