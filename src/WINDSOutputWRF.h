#pragma once

#include <string>

#include "WINDSGeneralData.h"
#include "WRFInput.h"
#include "QESNetCDFOutput.h"

/* Specialized output classes derived from QESNetCDFOutput for
   face center data (used for turbulence,...)
*/
class WINDSOutputWRF : public QESNetCDFOutput
{
public:
    WINDSOutputWRF()
        : QESNetCDFOutput()
    {}

    WINDSOutputWRF(WINDSGeneralData*, WRFInput *wrfInputData);

    ~WINDSOutputWRF() {}

    //save function be call outside
    void save(float);

private:

    WINDSGeneralData* WGD_;
    WRFInput *wrf_;
};
