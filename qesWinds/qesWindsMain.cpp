#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "QESNetCDFOutput.h"

#include "handleWINDSArgs.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"
#include "WINDSOutputVisualization.h"
#include "WINDSOutputWorkspace.h"

#include "WINDSOutputWRF.h"

#include "TURBGeneralData.h"
#include "TURBOutput.h"

#include "Solver.h"
#include "CPUSolver.h"
#include "DynamicParallelism.h"
#include "GlobalMemory.h"
#include "SharedMemory.h"

#include "Sensor.h"

namespace pt = boost::property_tree;

using namespace boost::gregorian;
using namespace boost::posix_time;


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

  if (arguments.fireMode) {
    outputVec.push_back(new WINDSOutputWRF(WGD, WID->simParams->wrfInputData));
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

  int numIterations = 5;
  int tempMaxIter = WID->simParams->maxIterations;

  for (int index = 0; index < WGD->totalTimeIncrements; index++) {
    // print time progress (time stamp and percentage)
    WGD->printTimeProgress(index);

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
      /*int ii = 208;
      int j = 318;
      int k = 115;
      int icell_face = ii + j * WGD->nx + k * WGD->nx * WGD->ny;
      int icell_cent = ii + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
      std::cout << "WGD->icellflag[icell_cent]:   " << WGD->icellflag[icell_cent] << std::endl;
      std::cout << "WGD->icellflag[icell_cent-1]:   " << WGD->icellflag[icell_cent - 1] << std::endl;
      std::cout << "WGD->icellflag[icell_cent+1]:   " << WGD->icellflag[icell_cent + 1] << std::endl;
      std::cout << "WGD->icellflag[icell_cent-(WGD->nx - 1)]:   " << WGD->icellflag[icell_cent - (WGD->nx - 1)] << std::endl;
      std::cout << "WGD->icellflag[icell_cent+(WGD->nx - 1)]:   " << WGD->icellflag[icell_cent + (WGD->nx - 1)] << std::endl;
      std::cout << "WGD->icellflag[icell_cent-(WGD->nx - 1) * (WGD->ny - 1)]:   " << WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] << std::endl;
      std::cout << "WGD->icellflag[icell_cent+(WGD->nx - 1) * (WGD->ny - 1)]:   " << WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] << std::endl;

      std::cout << "WGD->u0[icell_face]:   " << WGD->u0[icell_face] << std::endl;
      std::cout << "WGD->u0[icell_face+1]:   " << WGD->u0[icell_face + 1] << std::endl;
      std::cout << "WGD->v0[icell_face]:   " << WGD->v0[icell_face] << std::endl;
      std::cout << "WGD->v0[icell_face+WGD->nx]:   " << WGD->v0[icell_face + WGD->nx] << std::endl;
      std::cout << "WGD->w0[icell_face]:   " << WGD->w0[icell_face] << std::endl;
      std::cout << "WGD->w0[icell_face+WGD->nx * WGD->ny]:   " << WGD->w0[icell_face + WGD->nx * WGD->ny] << std::endl;*/
      solver->solve(WID, WGD, !arguments.solveWind);

      solver->lambda_old = solver->lambda;
      WGD->u0 = WGD->u;
      WGD->v0 = WGD->v;
      WGD->w0 = WGD->w;
      /*std::cout << "solver->lambda[icell_cent]:   " << solver->lambda[icell_cent] << std::endl;
      std::cout << "solver->lambda[icell_cent-1]:   " << solver->lambda[icell_cent - 1] << std::endl;
      std::cout << "solver->lambda[icell_cent+1]:   " << solver->lambda[icell_cent + 1] << std::endl;
      std::cout << "solver->lambda[icell_cent-(WGD->nx - 1)]:   " << solver->lambda[icell_cent - (WGD->nx - 1)] << std::endl;
      std::cout << "solver->lambda[icell_cent+(WGD->nx - 1)]:   " << solver->lambda[icell_cent + (WGD->nx - 1)] << std::endl;
      std::cout << "solver->lambda[icell_cent-(WGD->nx - 1) * (WGD->ny - 1)]:   " << solver->lambda[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] << std::endl;
      std::cout << "solver->lambda[icell_cent+(WGD->nx - 1) * (WGD->ny - 1)]:   " << solver->lambda[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] << std::endl;

      std::cout << "solver->R[icell_cent]:   " << solver->R[icell_cent] << std::endl;
      std::cout << "solver->R[icell_cent-1]:   " << solver->R[icell_cent - 1] << std::endl;
      std::cout << "solver->R[icell_cent+1]:   " << solver->R[icell_cent + 1] << std::endl;
      std::cout << "solver->R[icell_cent-(WGD->nx - 1)]:   " << solver->R[icell_cent - (WGD->nx - 1)] << std::endl;
      std::cout << "solver->R[icell_cent+(WGD->nx - 1)]:   " << solver->R[icell_cent + (WGD->nx - 1)] << std::endl;
      std::cout << "solver->R[icell_cent-(WGD->nx - 1) * (WGD->ny - 1)]:   " << solver->R[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] << std::endl;
      std::cout << "solver->R[icell_cent+(WGD->nx - 1) * (WGD->ny - 1)]:   " << solver->R[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] << std::endl;*/

      WGD->wall->wallLogBC(WGD);
      for (int i = 0; i < numIterations; i++) {
        WID->simParams->maxIterations = 500;
        /*std::cout << "i:   " << i << std::endl;
        std::cout << "WGD->u0[icell_face]:   " << WGD->u0[icell_face] << std::endl;
        std::cout << "WGD->u0[icell_face+1]:   " << WGD->u0[icell_face + 1] << std::endl;
        std::cout << "WGD->v0[icell_face]:   " << WGD->v0[icell_face] << std::endl;
        std::cout << "WGD->v0[icell_face+WGD->nx]:   " << WGD->v0[icell_face + WGD->nx] << std::endl;
        std::cout << "WGD->w0[icell_face]:   " << WGD->w0[icell_face] << std::endl;
        std::cout << "WGD->w0[icell_face+WGD->nx * WGD->ny]:   " << WGD->w0[icell_face + WGD->nx * WGD->ny] << std::endl;

        std::cout << "WGD->e[icell_cent]:   " << WGD->e[icell_cent] << std::endl;
        std::cout << "WGD->f[icell_cent]:   " << WGD->f[icell_cent] << std::endl;
        std::cout << "WGD->g[icell_cent]:   " << WGD->g[icell_cent] << std::endl;
        std::cout << "WGD->h[icell_cent]:   " << WGD->h[icell_cent] << std::endl;
        std::cout << "WGD->m[icell_cent]:   " << WGD->m[icell_cent] << std::endl;
        std::cout << "WGD->n[icell_cent]:   " << WGD->n[icell_cent] << std::endl;*/
        /*solver->lambda.clear();
        solver->lambda_old.clear();
        solver->lambda.resize(WGD->numcell_cent, 0.0);
        solver->lambda_old.resize(WGD->numcell_cent, 0.0);
        solver->R.clear();
        solver->R.resize(WGD->numcell_cent, 0.0);*/
        // Run WINDS simulation code
        solver->solve(WID, WGD, !arguments.solveWind);

        solver->lambda_old = solver->lambda;
        WGD->u0 = WGD->u;
        WGD->v0 = WGD->v;
        WGD->w0 = WGD->w;
        /*std::cout << "solver->lambda[icell_cent]:   " << solver->lambda[icell_cent] << std::endl;
        std::cout << "solver->lambda[icell_cent-1]:   " << solver->lambda[icell_cent - 1] << std::endl;
        std::cout << "solver->lambda[icell_cent+1]:   " << solver->lambda[icell_cent + 1] << std::endl;
        std::cout << "solver->lambda[icell_cent-(WGD->nx - 1)]:   " << solver->lambda[icell_cent - (WGD->nx - 1)] << std::endl;
        std::cout << "solver->lambda[icell_cent+(WGD->nx - 1)]:   " << solver->lambda[icell_cent + (WGD->nx - 1)] << std::endl;
        std::cout << "solver->lambda[icell_cent-(WGD->nx - 1) * (WGD->ny - 1)]:   " << solver->lambda[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] << std::endl;
        std::cout << "solver->lambda[icell_cent+(WGD->nx - 1) * (WGD->ny - 1)]:   " << solver->lambda[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] << std::endl;

        std::cout << "solver->R[icell_cent]:   " << solver->R[icell_cent] << std::endl;
        std::cout << "solver->R[icell_cent-1]:   " << solver->R[icell_cent - 1] << std::endl;
        std::cout << "solver->R[icell_cent+1]:   " << solver->R[icell_cent + 1] << std::endl;
        std::cout << "solver->R[icell_cent-(WGD->nx - 1)]:   " << solver->R[icell_cent - (WGD->nx - 1)] << std::endl;
        std::cout << "solver->R[icell_cent+(WGD->nx - 1)]:   " << solver->R[icell_cent + (WGD->nx - 1)] << std::endl;
        std::cout << "solver->R[icell_cent-(WGD->nx - 1) * (WGD->ny - 1)]:   " << solver->R[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] << std::endl;
        std::cout << "solver->R[icell_cent+(WGD->nx - 1) * (WGD->ny - 1)]:   " << solver->R[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] << std::endl;*/

        WGD->wall->wallLogBC(WGD);
        /*std::cout << "WGD->u0[icell_face]:   " << WGD->u0[icell_face] << std::endl;
        std::cout << "WGD->u0[icell_face+1]:   " << WGD->u0[icell_face + 1] << std::endl;
        std::cout << "WGD->v0[icell_face]:   " << WGD->v0[icell_face] << std::endl;
        std::cout << "WGD->v0[icell_face+WGD->nx]:   " << WGD->v0[icell_face + WGD->nx] << std::endl;
        std::cout << "WGD->w0[icell_face]:   " << WGD->w0[icell_face] << std::endl;
        std::cout << "WGD->w0[icell_face+WGD->nx * WGD->ny]:   " << WGD->w0[icell_face + WGD->nx * WGD->ny] << std::endl;*/
      }
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
      TGD->run(WGD);

    // /////////////////////////////
    // Output the various files requested from the simulation run (netcdf wind velocity, icell values, etc...)
    // /////////////////////////////
    for (auto id_out = 0u; id_out < outputVec.size(); id_out++) {
      outputVec.at(id_out)->save(WGD->timestamp[index]);
    }

    WGD->u0.clear();
    WGD->v0.clear();
    WGD->w0.clear();

    WGD->u.clear();
    WGD->v.clear();
    WGD->w.clear();

    WGD->u0.resize(WGD->numcell_face, 0.0);
    WGD->v0.resize(WGD->numcell_face, 0.0);
    WGD->w0.resize(WGD->numcell_face, 0.0);

    WGD->u.resize(WGD->numcell_face, 0.0);
    WGD->v.resize(WGD->numcell_face, 0.0);
    WGD->w.resize(WGD->numcell_face, 0.0);
  }

  // /////////////////////////////
  exit(EXIT_SUCCESS);
}
