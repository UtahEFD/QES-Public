#include "LocalMixingNetCDF.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

void LocalMixingNetCDF::defineMixingLength(const WINDSInputData* WID,WINDSGeneralData* WGD)
{
    // open NetCDF file (constructor)
    NetCDFInput* mixLengthInput;
    mixLengthInput = new NetCDFInput(WID->turbParams->filename);

    int nx_f,ny_f,nz_f;

    // nx,ny,ny from file
    mixLengthInput->getDimensionSize("x",nx_f);
    mixLengthInput->getDimensionSize("y",ny_f);
    mixLengthInput->getDimensionSize("z",nz_f);

    if(nx_f != WGD->nx-1 || ny_f != WGD->ny-1 || nz_f != WGD->nz-1) {
        std::cout << "[ERROR] \t domain size error in " << WID->turbParams->filename <<std::endl;
        exit(EXIT_FAILURE);
    }

    //access variable (to check if exist)
    NcVar NcVar_mixlength;
    mixLengthInput->getVariable(WID->turbParams->varname, NcVar_mixlength);

    if(!NcVar_mixlength.isNull()) { // => mixlength in NetCDF file
        // netCDF variables
        std::vector<size_t> start;
        std::vector<size_t> count;
        start = {0,0,0};
        count = {static_cast<unsigned long>(nz_f),
                 static_cast<unsigned long>(ny_f),
                 static_cast<unsigned long>(nx_f)};

        //read in mixilength
        mixLengthInput->getVariableData(WID->turbParams->varname,start,count,WGD->mixingLengths);
    } else {
        std::cout << "[ERROR] \t no field " << WID->turbParams->varname << " in "
                  << WID->turbParams->filename <<std::endl;
        exit(EXIT_FAILURE);
    }

    return;
}
