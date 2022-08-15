#include <iostream>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "util/QESNetCDFOutput.h"

#include "winds/handleWINDSArgs.h"

#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"
#include "winds/WINDSOutputVisualization.h"
#include "winds/WINDSOutputWorkspace.h"

#include "winds/TURBGeneralData.h"
#include "winds/TURBOutput.h"

#include "winds/Solver.h"
#include "winds/CPUSolver.h"
#include "winds/DynamicParallelism.h"
#include "winds/GlobalMemory.h"
#include "winds/SharedMemory.h"

#include "winds/Sensor.h"

int main(int argc, char *argv[])
{
  // QES-Winds - Version output information
  std::string Revision = "0";
  std::cout << "QES-Winds "
            << "1.0.0" << std::endl;
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
  //WINDSInputData* WID = parseXMLTree(arguments.quicFile);
  WINDSInputData *WID = new WINDSInputData(arguments.qesFile);
  if (!WID) {
    std::cerr << "[ERROR] QES Input file: " << arguments.qesFile << " not able to be read successfully." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Checking if
  if (arguments.compTurb && !WID->turbParams) {
    std::cerr << "[ERROR] Turbulence model is turned on without turbParams in QES Intput file "
              << arguments.qesFile << std::endl;
    exit(EXIT_FAILURE);
  }


  if (arguments.terrainOut) {
    if (WID->simParams->DTE_heightField) {
      std::cout << "Creating terrain OBJ....\n";
      WID->simParams->DTE_heightField->outputOBJ(arguments.filenameTerrain);
      std::cout << "OBJ created....\n";
    } else {
      std::cerr << "[ERROR] No dem file specified as input\n";
      return -1;
    }
  }

  // Generate the general WINDS data from all inputs
  WINDSGeneralData *WGD = new WINDSGeneralData(WID, arguments.solveType);

  // create WINDS output classes
  std::vector<QESNetCDFOutput *> outputVec;
  if (arguments.visuOutput) {
    outputVec.push_back(new WINDSOutputVisualization(WGD, WID, arguments.netCDFFileVisu));
  }
  if (arguments.wkspOutput) {
    outputVec.push_back(new WINDSOutputWorkspace(WGD, arguments.netCDFFileWksp));
  }

  // Generate the general TURB data from WINDS data
  // based on if the turbulence output file is defined
  TURBGeneralData *TGD = nullptr;
  if (arguments.compTurb) {
    TGD = new TURBGeneralData(WID, WGD);
  }
  if (arguments.compTurb && arguments.turbOutput) {
    outputVec.push_back(new TURBOutput(TGD, arguments.netCDFFileTurb));
  }

  // //////////////////////////////////////////
  //
  // Run the QES-Winds Solver
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

  int numIterations = 1;
  int tempMaxIter = WID->simParams->maxIterations;

  for (int index = 0; index < WGD->totalTimeIncrements; index++) {
    // print time progress (time stamp and percentage)
    if (!WID->simParams->wrfCoupling) {
      WGD->printTimeProgress(index);
    }
    // Reset icellflag values
    WGD->resetICellFlag();

    // Create initial velocity field from the new sensors
    WGD->applyWindProfile(WID, index, arguments.solveType);

    // Apply parametrizations
    WGD->applyParametrizations(WID);

    solver->lambda.clear();
    solver->lambda_old.clear();
    solver->lambda.resize(WGD->numcell_cent, 0.0);
    solver->lambda_old.resize(WGD->numcell_cent, 0.0);
    solver->R.clear();
    solver->R.resize(WGD->numcell_cent, 0.0);

    // Applying the log law and solver iteratively
    if (WID->simParams->logLawFlag == 1) {
      WID->simParams->maxIterations = tempMaxIter;
      solver->solve(WID, WGD, !arguments.solveWind);
      solver->lambda_old = solver->lambda;

      /*for (int i = 0; i < numIterations; i++) {
        WID->simParams->maxIterations = 500;
        WGD->wall->wallLogBC(WGD, false);
        // Run WINDS simulation code
        solver->solve(WID, WGD, !arguments.solveWind);

        solver->lambda_old = solver->lambda;
      }*/
      WGD->u0 = WGD->u;
      WGD->v0 = WGD->v;
      WGD->w0 = WGD->w;
      WGD->wall->wallLogBC(WGD, true);
      WGD->u = WGD->u0;
      WGD->v = WGD->v0;
      WGD->w = WGD->w0;
    } else {
      // Run WINDS simulation code
      solver->solve(WID, WGD, !arguments.solveWind);
    }

    std::cout << "Solver done!\n";

    // Run turbulence
    if (TGD != nullptr)
      TGD->run();

    // /////////////////////////////
    // WRF Coupling
    // /////////////////////////////
    if (WID->simParams->wrfCoupling) {
      // send our stuff to wrf input file
      std::cout << "Writing data back to the WRF file." << std::endl;
      WID->simParams->wrfInputData->extractWind(WGD);
    }

    // /////////////////////////////
    // Output the various files requested from the simulation run (netcdf wind velocity, icell values, etc...)
    // /////////////////////////////
    for (auto id_out = 0u; id_out < outputVec.size(); id_out++) {
      outputVec.at(id_out)->save(WGD->timestamp[index]);
    }

    if (WID->simParams->wrfCoupling) {
      // Re-read WRF data -- get new stuff from wrf input file... sync...
      std::cout << "Attempting to re-read data from WRF." << std::endl;
      WID->simParams->wrfInputData->updateFromWRF();
    }
  }

  if (WID->simParams->wrfCoupling)
    WID->simParams->wrfInputData->endWRFSession();

  // /////////////////////////////
  std::cout << "QES-Winds Exiting." << std::endl;
  exit(EXIT_SUCCESS);
}
