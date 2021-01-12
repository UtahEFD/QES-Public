#include "WINDSOutputWRF.h"

WINDSOutputWRF::WINDSOutputWRF(WINDSGeneralData *WGD, WRFInput *wrfInputData)
    : QESNetCDFOutput(), WGD_( WGD ), wrf_(wrfInputData)
{
    std::cout<<"[Output] \t Writing WIND Field back to WRF Input File" <<std::endl;
}


// Save output at cell-centered values
void WINDSOutputWRF::save(float timeOut)
{
    wrf_->extractWind( WGD_ );
};
