#include "LocalMixing.h"

// These take care of the circular reference
#include "URBInputData.h"
#include "URBGeneralData.h"

void LocalMixing::saveMixingLength(const URBInputData* UID,URBGeneralData* UGD)
{
    mixLengthOut=new NetCDFOutput(UID->localMixingParam->filename);

    NcDim NcDim_x=mixLengthOut->addDimension("x",UGD->nx-1);
    NcDim NcDim_y=mixLengthOut->addDimension("y",UGD->ny-1);
    NcDim NcDim_z=mixLengthOut->addDimension("z",UGD->nz-1);
    
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;
    
    mixLengthOut->addField(UID->localMixingParam->varname,"m","distance to nearest object",
                           {NcDim_z,NcDim_y,NcDim_z},ncFloat);
    
    mixLengthOut->saveField2D(UID->localMixingParam->varname,vector_index,vector_size,UGD->mixingLengths);
    
    return;
}
