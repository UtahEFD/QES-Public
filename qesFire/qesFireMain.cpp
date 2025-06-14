/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Matthew Moody
 * Copyright (c) 2025 Jeremy Gibbs
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Brian Bailey
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the Lic
 * ense.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
#include <iostream>
#include <netcdf>
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <chrono>

#include "util/calcTime.h"
#include "util/NetCDFInput.h"
#include "util/ParseException.h"
#include "util/ParseInterface.h"
#include "util/QESout.h"
#include "util/QESNetCDFOutput.h"

#include "qes/Domain.h"

#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"
#include "winds/WINDSOutputVisualization.h"
#include "winds/WINDSOutputWorkspace.h"
#include "winds/TURBGeneralData.h"
#include "winds/TURBOutput.h"
#include "winds/Solver.h"
#include "winds/SolverFactory.h"
#include "winds/Sensor.h"

#include "plume/PLUMEInputData.h"
#include "plume/PLUMEGeneralData.h"
#include "plume/ParticleOutput.h"

#include "fire/Fire.h"
#include "fire/FIREOutput.h"
#include "fire/SourceFire.h"
#include "fire/Smoke.h"

#include "handleQESArgs.h"

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
    QESout::error("QES Input file: " + arguments.qesWindsParamFile + " not able to be read successfully.");
  }

  // Checking if
  if (arguments.compTurb && !WID->turbParams) {
    QESout::error("Turbulence model is turned on without turbParams in QES Intput file "
                  + arguments.qesWindsParamFile);
  }

  qes::Domain domain(WID->simParams->domain, WID->simParams->grid);

  // Generate the general WINDS data from all inputs
  WINDSGeneralData *WGD = new WINDSGeneralData(WID, domain, arguments.solveType);

  // /////////////////////////////
  //
  // Run Fire Code
  //
  // /////////////////////////////
  bool GPUFLAG = false;
#ifdef HAS_CUDA
  if (arguments.solveType == DYNAMIC_P || arguments.solveType == Global_M || arguments.solveType == Shared_M) {
    GPUFLAG = true;
  }
#endif

  // Generate fire general data
  Fire *fire = new Fire(WID, WGD, GPUFLAG);

  /**
   * Set fuel map
   **/
  fire->FuelMap(WID, WGD);

  // Create FIREOutput manager
  std::vector<QESNetCDFOutput *> outFire;
  outFire.push_back(new FIREOutput(WGD, fire, arguments.netCDFFileFireOut));


  // //////////////////////////////////////////
  // Set the QES-Winds Solver
  SolverFactory solverFactory;
  Solver *solver = solverFactory.create(arguments.solveType, WGD->domain, WID->simParams->tolerance);

  /**
   * Time variables to track fire time and sensor timesteps
   **/
  QEStime simTimeStart = WGD->timestamp[0];
  QEStime simTimeCurr = simTimeStart;

  /**
   * Temporay velocity arrays to hold initial wind field data per sensor time series
   **/
  std::vector<float> Fu0(domain.numFaceCentered(), 0.0);
  std::vector<float> Fv0(domain.numFaceCentered(), 0.0);
  std::vector<float> Fw0(domain.numFaceCentered(), 0.0);

  // Generate the general TURB data from WINDS data
  TURBGeneralData *TGD = nullptr;
  if (arguments.compTurb) {
    TGD = new TURBGeneralData(WID, WGD);
  }
  if (arguments.compTurb && arguments.turbOutput) {
    outFire.push_back(new TURBOutput(TGD, arguments.netCDFFileTurb));
  }

  // PLUME
  PlumeInputData *PID = nullptr;
  PLUMEGeneralData *PGD = nullptr;
  // Smoke *smoke = nullptr;
  if (arguments.compPlume) {
    PID = new PlumeInputData(arguments.qesPlumeParamFile);
    if (!PID)
      QESout::error("QES-Plume input file: " + arguments.qesPlumeParamFile + " not able to be read successfully.");
    // Create instance of Plume model class
    PGD = new PLUMEGeneralData(arguments.plumeParameters, PID, WGD, TGD);
    // Create the particle model for smoke
    PGD->addParticleModel(new ParticleModel("smoke"));
  }

  /**
   * Loop  for sensor time data
   **/

  for (int index = 0; index < WGD->totalTimeIncrements; index++) {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "New Sensor Data" << std::endl;
    std::cout << "index = " << index << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    /**
     * Reset icellflag values
     **/
    WGD->resetICellFlag();

    /**
     * Create initial velocity field from the new sensors
     **/
    WGD->applyWindProfile(WID, index, arguments.solveType);

    /**
     * Apply parametrizations
     **/
    WGD->applyParametrizations(WID);

    /**
     * Run WINDS simulation code
     **/
    solver->solve(WGD, WID->simParams->maxIterations);

    /**
     * Run turbulence if specified
     **/

    if (TGD != nullptr) TGD->run();

    /**
     * Save initial fields from sensor time to reset after each time+fire loop
     **/
    Fu0 = WGD->u0;
    Fv0 = WGD->v0;
    Fw0 = WGD->w0;

    simTimeCurr = WGD->timestamp[index];///< Simulation time for current sensor time
    QEStime endtime;///< End time for fire time loop
    if (WGD->totalTimeIncrements == 1) {
      endtime = WGD->timestamp[index] + WID->fires->fireDur;
    } else if (index == WGD->totalTimeIncrements - 1) {
      endtime = simTimeStart + WID->fires->fireDur;
    } else {
      endtime = WGD->timestamp[index + 1];
    }

    /**
     * Fire time loop for current sensor time
     **/

    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "[QES-Fire]\t Fire simulation stating from " << simTimeCurr << " to " << endtime << "." << std::endl;
    while (simTimeCurr < endtime) {
      auto startTimeFire = std::chrono::high_resolution_clock::now();

      /**
       * load initial velocity for current sensor series
       */
      WGD->u0 = Fu0;
      WGD->v0 = Fv0;
      WGD->w0 = Fw0;

      // Run fire induced winds (default) if flag is not set in command line
      if (!arguments.fireWindsFlag) {
        std::cout << "-------------------------------------------------------------------" << std::endl;
        /**
         * Run ROS model to get initial spread rate and fire properties
         */
        fire->LevelSetNB(WGD);

        /**
         * Calculate fire-induced winds from burning cells
         */
        fire->potential(WGD);
      }
      std::cout << "-------------------------------------------------------------------" << std::endl;
      /**
       * Apply parameterizations
       */
      WGD->applyParametrizations(WID);

      /**
       * Run run wind solver to calculate mass conserved velocity field including fire-induced winds
       */
      solver->solve(WGD, WID->simParams->maxIterations);
      if (TGD != nullptr) TGD->run();

      std::cout << "-------------------------------------------------------------------" << std::endl;
      /**
       * Run ROS model to calculate spread rates with updated winds
       */
      fire->LevelSetNB(WGD);

      /**
       * Advance fire front through level set method
       */
      fire->move(WGD);

      if (PGD != nullptr) {
        // std::cout << "------Running Plume------" << std::endl;
        //  Loop through domain to find new smoke sources
        for (int j = 1; j < domain.ny() - 2; j++) {
          for (int i = 1; i < domain.nx() - 2; i++) {
            int idx = i + j * (domain.nx() - 1);
            // If smoke flag set in fire program, get x, y, z location and set source
            if (fire->smoke_flag[idx] == 1) {
              float x_pos = i * domain.dx();
              float y_pos = j * domain.dy();
              float z_pos = WGD->terrain[idx] + 1;
              int ppt = 20;
              FireSourceBuilder FSB;
              FSB.setSourceParam({ x_pos, y_pos, z_pos }, simTimeCurr, simTimeCurr + fire->fire_cells[idx].properties.tau, ppt);
              // Add source to plume
              PGD->models["smoke"]->addSource(FSB.create());
              // Clear smoke flag in fire program so no duplicate source set next time step
              fire->smoke_flag[idx] = 0;
            }
          }
        }
        // std::cout << "Plume run" << std::endl;
        QEStime pendtime = simTimeCurr + fire->dt;///< End time for fire time loop run until end of fire timestep
        PGD->run(pendtime, WGD, TGD);
        // std::cout << "------Plume Finished------" << std::endl;
      }

      /**
       * Advance fire time from variable fire timestep
       */
      simTimeCurr += fire->dt;

      auto endTimerFire = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = endTimerFire - startTimeFire;

      std::cout << "-------------------------------------------------------------------\n"
                << "[QES-Fire]\t Fire step completed.\n"
                << "[QES-Fire]\t Current time = " << simTimeCurr << "\n"
                << "\t\t elapsed time: " << elapsed.count() << " s" << std::endl;

      /**
       * Save fire data to netCDF file
       */
      for (auto &out : outFire) {
        out->save(simTimeCurr);
      }
    }
  }
  std::cout << "Simulation finished" << std::endl;

  delete WID;
  delete WGD;
  delete TGD;
  delete PID;
  delete PGD;
  for (auto p : outFire) {
    delete p;
  }


  exit(EXIT_SUCCESS);
}
