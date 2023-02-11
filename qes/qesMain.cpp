/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "util/QESout.h"

#include "util/QESNetCDFOutput.h"

#include "handleQESArgs.h"

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

//#include "Args.hpp"
#include "plume/PlumeInputData.hpp"
#include "plume/Plume.hpp"
#include "plume/PlumeOutput.h"
#include "plume/PlumeOutputParticleData.h"


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
      WID->simParams->DTE_heightField->outputOBJ(arguments.filenameTerrain);
      std::cout << "OBJ created....\n";
    } else {
      QESout::error("No dem file specified as input");
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

  Plume *plume = nullptr;
  // create output instance
  std::vector<QESNetCDFOutput *> outputPlume;

  if (arguments.compPlume) {
    // Create instance of Plume model class
    plume = new Plume(PID, WGD, TGD);

    // always supposed to output lagrToEulOutput data
    outputPlume.push_back(new PlumeOutput(PID, plume, arguments.outputPlumeFile));
    if (arguments.doParticleDataOutput == true) {
      outputPlume.push_back(new PlumeOutputParticleData(PID, plume, arguments.outputParticleDataFile));
    }
  }

  // //////////////////////////////////////////
  //
  // Run the QES-Winds Solver
  //
  // //////////////////////////////////////////
  Solver *solver = setSolver(arguments.solveType, WID, WGD);

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
    solver->solve(WID, WGD, !arguments.solveWind);

    std::cout << "Solver done!\n";

    // Run turbulence
    if (TGD != nullptr) {
      TGD->run();
    }

    // /////////////////////////////
    // Output the various files requested from the simulation run
    // (netcdf wind velocity, icell values, etc...
    // /////////////////////////////
    for (auto outItr = outputVec.begin(); outItr != outputVec.end(); ++outItr) {
      (*outItr)->save(WGD->timestamp[index]);
    }

    // Run plume advection model
    if (plume != nullptr) {
      QEStime endtime;
      if (WGD->totalTimeIncrements == 1) {
        endtime = WGD->timestamp[index] + PID->plumeParams->simDur;
      } else if (index == WGD->totalTimeIncrements - 1) {
        endtime = WGD->timestamp[index] + (WGD->timestamp[index] - WGD->timestamp[index - 1]);
      } else {
        endtime = WGD->timestamp[index + 1];
      }
      plume->run(endtime, WGD, TGD, outputPlume);
    }
  }

  if (plume != nullptr) {
    std::cout << "[QES-Plume] \t Finished. \n";
    std::cout << "End run particle summary \n";
    plume->showCurrentStatus();
  }

  exit(EXIT_SUCCESS);
}

Solver *setSolver(const int solveType, WINDSInputData *WID, WINDSGeneralData *WGD)
{
  Solver *solver = nullptr;
  if (solveType == CPU_Type) {
    std::cout << "Run Serial Solver (CPU) ..." << std::endl;
    solver = new CPUSolver(WID, WGD);
#ifdef HAS_CUDA
  } else if (solveType == DYNAMIC_P) {
    std::cout << "Run Dynamic Parallel Solver (GPU) ..." << std::endl;
    solver = new DynamicParallelism(WID, WGD);
  } else if (solveType == Global_M) {
    std::cout << "Run Global Memory Solver (GPU) ..." << std::endl;
    solver = new GlobalMemory(WID, WGD);
  } else if (solveType == Shared_M) {
    std::cout << "Run Shared Memory Solver (GPU) ..." << std::endl;
    solver = new SharedMemory(WID, WGD);
#endif    
  } else {
    QESout::error("Invalid solve type");
  }
  return solver;
}
