/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES
 *
 * GPL-3.0 License
 *
 * QES is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

#include <iostream>

#include "util/QESout.h"

#include "util/QESNetCDFOutput.h"

#include "handleQESArgs.h"

#include "qes/Domain.h"

#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"
#include "winds/WINDSOutputVisualization.h"
#include "winds/WINDSOutputWorkspace.h"

#include "winds/TURBGeneralData.h"
#include "winds/TURBOutput.h"

#include "winds/Solver.h"
#include "winds/Solver_CPU.h"
#include "winds/Solver_CPU_RB.h"
#ifdef HAS_CUDA
#include "winds/Solver_GPU_DynamicParallelism.h"
#include "winds/Solver_GPU_GlobalMemory.h"
#include "winds/Solver_GPU_SharedMemory.h"
#endif

#include "winds/Sensor.h"

// #include "Args.hpp"
#include "plume/PLUMEInputData.h"
#include "plume/PLUMEGeneralData.h"


Solver *setSolver(const int, WINDSInputData *, WINDSGeneralData *);

int main(int argc, char *argv[])
{
  QESout::splashScreen();

  // ///////////////////////////////////
  // Parse Command Line arguments
  // ///////////////////////////////////

  // Command line arguments are processed in a uniform manner using
  // cross-platform code.  Check the WINDSArgs class for details on
  // how to extend the arguments.
  QESArgs arguments;
  arguments.processArguments(argc, argv);

  // ///////////////////////////////////
  // Read and Process any Input for the system
  // ///////////////////////////////////

  // Parse the base XML QUIC file -- contains simulation parameters
  WINDSInputData *WID = new WINDSInputData(arguments.qesWindsParamFile);
  if (!WID) {
    QESout::error("QES Input file: " + arguments.qesWindsParamFile
                  + " not able to be read successfully.");
  }
  // parse xml settings

  PlumeInputData *PID = nullptr;
  if (arguments.compPlume) PID = new PlumeInputData(arguments.qesPlumeParamFile);

  // Checking if
  if (arguments.compTurb && !WID->turbParams) {
    QESout::error("Turbulence model is turned on without turbParams in QES Intput file "
                  + arguments.qesWindsParamFile);
  }

  if (arguments.terrainOut) {
    if (WID->simParams->DTE_heightField) {
      std::cout << "Creating terrain OBJ....\n";
      WID->simParams->DTE_heightField->outputOBJ(arguments.outputFileBasename + "_terrainOut.obj");
      std::cout << "OBJ created....\n";
    } else {
      QESout::error("No dem file specified as input");
    }
  }
  qes::Domain domain(WID->simParams->domain[0], WID->simParams->domain[1], WID->simParams->domain[2], WID->simParams->grid[0], WID->simParams->grid[1], WID->simParams->grid[2]);

  // Generate the general WINDS data from all inputs
  WINDSGeneralData *WGD = new WINDSGeneralData(WID, domain, arguments.solveType);

  // create WINDS output classes
  std::vector<QESNetCDFOutput *> outputVec;
  if (arguments.visuOutput) {
    outputVec.push_back(new WINDSOutputVisualization(WGD, WID, arguments.outputFileBasename + "_windsOut.nc"));
  }
  if (arguments.wkspOutput) {
    outputVec.push_back(new WINDSOutputWorkspace(WGD, arguments.outputFileBasename + "_windsWk.nc"));
  }

  // Generate the general TURB data from WINDS data
  // based on if the turbulence output file is defined
  TURBGeneralData *TGD = nullptr;
  if (arguments.compTurb) {
    TGD = new TURBGeneralData(WID, WGD);
  }
  if (arguments.compTurb && arguments.turbOutput) {
    outputVec.push_back(new TURBOutput(TGD, arguments.outputFileBasename + "_turbOut.nc"));
  }

  PLUMEGeneralData *PGD = nullptr;
  if (arguments.compPlume) {
    // Create instance of Plume model class
    PGD = new PLUMEGeneralData(arguments.plumeParameters, PID, WGD, TGD);
  }

  // Set the QES-Winds Solver
  Solver *solver = setSolver(arguments.solveType, WID, WGD);
  if (!solver) { QESout::error("Invalid solver"); }

  for (int index = 0; index < WGD->totalTimeIncrements; index++) {
    // print time progress (time stamp and percentage)
    WGD->printTimeProgress(index);

    // Reset icellflag values
    WGD->resetICellFlag();

    // Create initial velocity field from the new sensors
    WGD->applyWindProfile(WID, index, arguments.solveType);

    // Apply parametrizations
    WGD->applyParametrizations(WID);

    // Run WINDS simulation code
    solver->solve(WGD, WID->simParams->maxIterations);

    // Run turbulence
    if (TGD != nullptr) {
      TGD->run();
    }

    // /////////////////////////////
    // Output the various files requested from the simulation run
    // (netcdf wind velocity, icell values, etc...
    // /////////////////////////////
    for (auto &out : outputVec) {
      out->save(WGD->timestamp[index]);
    }

    // Run plume advection model
    if (PGD != nullptr) {
      QEStime endTime = WGD->nextTimeInstance(index, PID->plumeParams->simDur);
      PGD->run(endTime, WGD, TGD);
    }
  }

  if (PGD != nullptr) {
    PGD->showCurrentStatus();
  }

  delete WID;
  delete WGD;
  delete TGD;
  for (auto p : outputVec) {
    delete p;
  }

  delete PID;
  delete PGD;

  exit(EXIT_SUCCESS);
}

Solver *setSolver(const int solveType, WINDSInputData *WID, WINDSGeneralData *WGD)
{
  Solver *solver = nullptr;
  if (solveType == CPU_Type) {
#ifdef _OPENMP
    solver = new Solver_CPU_RB(WGD->domain, WID->simParams->tolerance);
#else
    solver = new Solver_CPU(WGD->domain, WID->simParams->tolerance);
#endif

#ifdef HAS_CUDA
  } else if (solveType == DYNAMIC_P) {
    solver = new Solver_GPU_DynamicParallelism(WGD->domain, WID->simParams->tolerance);
  } else if (solveType == Global_M) {
    solver = new Solver_GPU_GlobalMemory(WGD->domain, WID->simParams->tolerance);
  } else if (solveType == Shared_M) {
    solver = new Solver_GPU_SharedMemory(WGD->domain, WID->simParams->tolerance);
#endif
  } else {
    QESout::error("Invalid solver type");
  }
  return solver;
}
