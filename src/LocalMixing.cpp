#include "LocalMixing.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

void LocalMixing::saveMixingLength(const WINDSInputData* WID,WINDSGeneralData* WGD)
{
    // open NetCDF file (constructor)
    mixLengthOut=new NetCDFOutput(WID->localMixingParam->filename);

    // create NcDimension for x,y,z (with ghost cell)
    NcDim NcDim_x=mixLengthOut->addDimension("x",WGD->nx-1);
    NcDim NcDim_y=mixLengthOut->addDimension("y",WGD->ny-1);
    NcDim NcDim_z=mixLengthOut->addDimension("z",WGD->nz-1);

    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;

    vector_index = { 0, 0, 0};
    vector_size  = { static_cast<unsigned long>(WGD->nz-1),
                     static_cast<unsigned long>(WGD->ny-1),
                     static_cast<unsigned long>(WGD->nx-1)};

    // create NetCDF filed in file
    mixLengthOut->addField(WID->localMixingParam->varname,"m","distance to nearest object",
                           {NcDim_z,NcDim_y,NcDim_x},ncFloat);

    // dump mixingLengths to file
    mixLengthOut->saveField2D(WID->localMixingParam->varname,vector_index,vector_size,WGD->mixingLengths);

    return;
}
