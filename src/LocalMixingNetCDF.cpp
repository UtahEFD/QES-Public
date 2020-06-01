#include "LocalMixingNetCDF.h"

// These take care of the circular reference
#include "URBInputData.h"
#include "URBGeneralData.h"

void LocalMixingNetCDF::defineMixingLength(const URBInputData* UID,URBGeneralData* UGD) 
{
    // open NetCDF file (constructor)
    NetCDFInput* mixLengthInput;
    mixLengthInput = new NetCDFInput(UID->localMixingParam->filename);

    int nx_f,ny_f,nz_f;
    
    // nx,ny,ny from file
    mixLengthInput->getDimensionSize("x",nx_f);
    mixLengthInput->getDimensionSize("y",ny_f);
    mixLengthInput->getDimensionSize("z",nz_f);
    
    if(nx_f != UGD->nx-1 || ny_f != UGD->ny-1 || nz_f != UGD->nz-1) {
        std::cout << "[ERROR] \t domain size error in " << UID->localMixingParam->filename <<std::endl;
        exit(EXIT_FAILURE);
    }
        
    //access variable (to check if exist)
    NcVar NcVar_mixlength;
    mixLengthInput->getVariable(UID->localMixingParam->varname, NcVar_mixlength);

    if(!NcVar_mixlength.isNull()) { // => mixlength in NetCDF file
        // netCDF variables
        std::vector<size_t> start;
        std::vector<size_t> count;
        start = {0,0,0};
        count = {static_cast<unsigned long>(nz_f),
                 static_cast<unsigned long>(ny_f),
                 static_cast<unsigned long>(nx_f)};
        
        //read in mixilength
        mixLengthInput->getVariableData(UID->localMixingParam->varname,start,count,UGD->mixingLengths);
    } else {
        std::cout << "[ERROR] \t no field " << UID->localMixingParam->varname << " in " 
                  << UID->localMixingParam->filename <<std::endl;
        exit(EXIT_FAILURE);
    }
    
    return;
}

