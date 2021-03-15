#include <iostream>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "QESNetCDFOutput.h"

#include "handleWINDSArgs.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"
#include "WINDSOutputVisualization.h"
#include "WINDSOutputWorkspace.h"

#include "TURBGeneralData.h"
#include "TURBOutput.h"

#include "Solver.h"
#include "CPUSolver.h"
#include "DynamicParallelism.h"
#include "GlobalMemory.h"
#include "SharedMemory.h"

#include "Sensor.h"

namespace bt=boost::posix_time;

int main(int argc, char *argv[])
{
    // QES-Winds - Version output information
    std::string Revision = "0";
    std::cout << "QES-Winds " << "1.0.0" << std::endl;
#ifdef HAS_OPTIX
    std::cout << "OptiX is available!" << std::endl;
#endif

    // ///////////////////////////////////
    // Parse Command Line arguments
    // ///////////////////////////////////

    // Command line arguments are processed in a uniform manner using
    // cross-platform code.  Check the WINDSArgs class for details on
    // how to extend the arguments.
    WINDSArgs arguments;
    arguments.processArguments(argc, argv);

    // ///////////////////////////////////
    // Read and Process any Input for the system
    // ///////////////////////////////////

    // Parse the base XML QUIC file -- contains simulation parameters
    WINDSInputData* WID = new WINDSInputData(arguments.quicFile);
    if ( !WID ) {
        std::cerr << "[ERROR] QUIC Input file: " << arguments.quicFile <<
            " not able to be read successfully." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Checking if
    if (arguments.compTurb && !WID->turbParams) {
        std::cerr << "[ERROR] Turbulence model is turned on without LocalMixingParam in QES Intput file "
                  << arguments.quicFile << std::endl;
        exit(EXIT_FAILURE);
    }

    if (arguments.terrainOut) {
        if (WID->simParams->DTE_heightField) {
            std::cout << "Creating terrain OBJ....\n";
            WID->simParams->DTE_heightField->outputOBJ(arguments.filenameTerrain);
            std::cout << "OBJ created....\n";
        }
        else {
            std::cerr << "[ERROR] No dem file specified as input\n";
            return -1;
        }
    }

    // Generate the general WINDS data from all inputs
    WINDSGeneralData* WGD = new WINDSGeneralData(WID, arguments.solveType);
    
    // create WINDS output classes
    std::vector<QESNetCDFOutput*> outputVec;
    if (arguments.visuOutput) {
        outputVec.push_back(new WINDSOutputVisualization(WGD,WID,arguments.netCDFFileVisu));
    }
    if (arguments.wkspOutput) {
        outputVec.push_back(new WINDSOutputWorkspace(WGD,arguments.netCDFFileWksp));
    }

    // Generate the general TURB data from WINDS data
    // based on if the turbulence output file is defined
    TURBGeneralData* TGD = nullptr;
    if (arguments.compTurb) {
        TGD = new TURBGeneralData(WID,WGD);
    }
    if (arguments.compTurb && arguments.turbOutput) {
        outputVec.push_back(new TURBOutput(TGD,arguments.netCDFFileTurb));
    }

    // //////////////////////////////////////////
    //
    // Run the CUDA-WINDS Solver
    //
    // //////////////////////////////////////////
    Solver *solver, *solverC = nullptr;
    if (arguments.solveType == CPU_Type) {
        std::cout << "Run Serial Solver (CPU) ..." << std::endl;
        solver = new CPUSolver(WID, WGD);
    } else if (arguments.solveType == DYNAMIC_P) {
        std::cout << "Run Dynamic Parallel Solver (GPU) ..." << std::endl;
        solver = new DynamicParallelism(WID, WGD);
    } else if (arguments.solveType == Global_M) {
        std::cout << "Run Global Memory Solver (GPU) ..." << std::endl;
        solver = new GlobalMemory(WID, WGD);
    } else if (arguments.solveType == Shared_M) {
        std::cout << "Run Shared Memory Solver (GPU) ..." << std::endl;
        solver = new SharedMemory(WID, WGD);
    } else {
        std::cerr << "[ERROR] invalid solve type\n";
        exit(EXIT_FAILURE);
    }

    //check for comparison
    if (arguments.compareType) {
        if (arguments.compareType == CPU_Type)
            solverC = new CPUSolver(WID, WGD);
        else if (arguments.compareType == DYNAMIC_P)
            solverC = new DynamicParallelism(WID, WGD);
        else if (arguments.compareType == Global_M)
            solverC = new GlobalMemory(WID, WGD);
        else if (arguments.compareType == Shared_M)
            solverC = new SharedMemory(WID, WGD);
        else {
            std::cerr << "[ERROR] invalid comparison type\n";
            exit(EXIT_FAILURE);
        }
    }

    for (int index = 0; index < WID->simParams->totalTimeIncrements; index++) {
        std::cout << "Running time step: " <<  bt::to_iso_extended_string(WGD->timestamp[index]) << std::endl;
        // Reset icellflag values
        //WGD->resetICellFlag();
        
        // Create initial velocity field from the new sensors
        WID->metParams->sensors[0]->inputWindProfile(WID, WGD, index, arguments.solveType);
        
        // Run WINDS simulation code
        solver->solve(WID, WGD, !arguments.solveWind );
        std::cout << "Solver done!\n";

        for (int k = 0; k < 1; ++k) { 
            // set u0,v0 to current solution    
            WGD->u0 = WGD->u;    
            WGD->v0 = WGD->v;
            WGD->w0 = WGD->w;
            
            // Apply parametrizations
            WGD->applyParametrizations(WID);
            
            // Run WINDS simulation code
            solver->solve(WID, WGD, !arguments.solveWind );
            std::cout << "Solver done!\n";
        }
        
        // Run turbulence
        if(TGD != nullptr) {
            TGD->run(WGD);
        }
        
        // /////////////////////////////
        // Output the various files requested from the simulation run
        // (netcdf wind velocity, icell values, etc...
        // /////////////////////////////
        for(auto id_out=0u;id_out<outputVec.size();id_out++) {
            outputVec.at(id_out)->save(WGD->timestamp[index]); // need to replace 0.0 with timestep
        }
    }

    // /////////////////////////////
    exit(EXIT_SUCCESS);
}
