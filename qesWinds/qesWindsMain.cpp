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
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
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

#include "util/QESNetCDFOutput.h"
#include "util/QESout.h"

#include "winds/handleWINDSArgs.h"

#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"
#include "winds/WINDSOutputVisualization.h"
#include "winds/WINDSOutputWorkspace.h"

#include "winds/TURBGeneralData.h"
#include "winds/TURBOutput.h"

#include "winds/Solver.h"
#include "winds/CPUSolver.h"
#ifdef HAS_CUDA
#include "winds/DynamicParallelism.h"
#include "winds/GlobalMemory.h"
#include "winds/SharedMemory.h"
#endif
#include "winds/Sensor.h"

int main(int argc, char *argv[])
{
  QESout::splashScreen();

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
  WINDSInputData *WID = new WINDSInputData(arguments.qesWindsParamFile);
  if (!WID) {
    QESout::error("QES Input file: " + arguments.qesWindsParamFile + " not able to be read successfully.");
  }

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

  // //////////////////////////////////////////
  //
  // Run the QES-Winds Solver
  //
  // //////////////////////////////////////////
  Solver *solver = nullptr;
  if (arguments.solveType == CPU_Type) {
    std::cout << "Run Serial Solver (CPU) ..." << std::endl;
    solver = new CPUSolver(WID, WGD);
#ifdef HAS_CUDA
  } else if (arguments.solveType == DYNAMIC_P) {
    std::cout << "Run Dynamic Parallel Solver (GPU) ..." << std::endl;
    solver = new DynamicParallelism(WID, WGD);
  } else if (arguments.solveType == Global_M) {
    std::cout << "Run Global Memory Solver (GPU) ..." << std::endl;
    solver = new GlobalMemory(WID, WGD);
  } else if (arguments.solveType == Shared_M) {
    std::cout << "Run Shared Memory Solver (GPU) ..." << std::endl;
    solver = new SharedMemory(WID, WGD);
#endif
  } else {
    QESout::error("Invalid solve type");
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

    solver->resetLambda();

    // Applying the log law and solver iteratively
    if (WID->simParams->logLawFlag == 1) {
      WID->simParams->maxIterations = tempMaxIter;
      solver->solve(WID, WGD, !arguments.solveWind);

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

    //std::cout << "Solver done!\n";

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
