/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
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


#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "util/calcTime.h"

#include "plume/PlumeInputData.hpp"
#include "util/NetCDFInput.h"
#include "plume/Plume.hpp"
#include "plume/PlumeOutput.h"
#include "plume/PlumeOutputParticleData.h"

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
#include "winds/Solver_CPU_RB.h"
#ifdef HAS_CUDA
#include "winds/DynamicParallelism.h"
#include "winds/GlobalMemory.h"
#include "winds/SharedMemory.h"
#endif
#include "winds/Sensor.h"

#include "fire/Fire.h"
#include "fire/FIREOutput.h"
#include "fire/SourceFire.h"
#include <chrono>
#include "fire/Smoke.h"

namespace pt = boost::property_tree;

using namespace boost::gregorian;
using namespace boost::posix_time;

using namespace netCDF;             //plume
using namespace netCDF::exceptions; //plume

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

  if (arguments.terrainOut) {
    if (WID->simParams->DTE_heightField) {
      std::cout << "Creating terrain OBJ....\n";
      WID->simParams->DTE_heightField->outputOBJ(arguments.filenameTerrain);
      std::cout << "OBJ created....\n";
    } else {
      QESout::error("No dem file specified as input");
    }
  }

  


  
  
  // //////////////////////////////////////////
  //
  // Run the QES-Winds Solver
  //
  // //////////////////////////////////////////
    Solver *solver = nullptr;
  if (arguments.solveType == CPU_Type) {
#ifdef _OPENMP
    std::cout << "Run Red/Black Solver (CPU) ..." << std::endl;
    solver = new Solver_CPU_RB(WID, WGD);
#else
    std::cout << "Run Serial Solver (CPU) ..." << std::endl;
    solver = new CPUSolver(WID, WGD);
#endif

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

  // /////////////////////////////
  //
  // Run Fire Code
  //
  // /////////////////////////////

  /** 
     * Create Fire Map
     **/

  Fire *fire = new Fire(WID, WGD);
  /**
   * Set fuel map
   */
  fire->FuelMap(WID, WGD);

  /**
     * Create FIREOutput manager
     **/

  std::vector<QESNetCDFOutput *> outFire;
  outFire.push_back(new FIREOutput(WGD, fire, arguments.netCDFFileFireOut));


  /**
     * Time variables to track fire time and sensor timesteps
     **/
  QEStime simTimeStart = WGD->timestamp[0];
  QEStime simTimeCurr = simTimeStart;

  //std::vector<float> Fu0;
  //std::vector<float> Fv0;
  //std::vector<float> Fw0;


  // Generate the general TURB data from WINDS data
  // based on if the turbulence output file is defined
  TURBGeneralData *TGD = nullptr;
  if (arguments.compTurb) {
    TGD = new TURBGeneralData(WID, WGD);
  }
  if (arguments.compTurb && arguments.turbOutput) {
    outputVec.push_back(new TURBOutput(TGD, arguments.netCDFFileTurb));
  }
  // parse Plume xml settings
  PlumeInputData *PID = nullptr;
  Plume *plume = nullptr;
  Smoke *smoke = nullptr;
  // create output instance
  std::vector<QESNetCDFOutput *> PoutputVec;
  if(arguments.compPlume){
    PID = new PlumeInputData(arguments.qesPlumeParamFile);
    if (!PID)
      QESout::error("QES-Plume input file: " + arguments.qesPlumeParamFile + " not able to be read successfully.");
    // Create instance of Plume model class
    plume = new Plume(PID, WGD, TGD);
    smoke = new Smoke();
    
    // always supposed to output lagrToEulOutput data
    PoutputVec.push_back(new PlumeOutput(PID, plume, arguments.outputPlumeFile));
    if (arguments.doParticleDataOutput == true) {
      PoutputVec.push_back(new PlumeOutputParticleData(PID, plume, arguments.outputParticleDataFile));
    }
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
    solver->solve(WID, WGD, !arguments.solveWind);

    std::cout << "Solver done!\n";

    /**
       * Run turbulence if specified
       **/
   
    if (TGD != nullptr) {
      TGD->run();
      std::cout << "Turbulance calculated\n";
    }
   
    /**
       * Save initial fields from sensor time to reset after each time+fire loop
       **/
    //Fu0 = WGD->u0;///< Initial u-velocity for sensor timestep
    //Fv0 = WGD->v0;///< Initial v-velocity for sensor timestep
    //Fw0 = WGD->w0;///< Initial w-velocity for sensor timestep


    simTimeCurr = WGD->timestamp[index];///< Simulation time for current sensor time
    QEStime endtime;///< End time for fire time loop
    if (WGD->totalTimeIncrements == 1) {
      endtime = WGD->timestamp[index] + WID->fires->fireDur;
    } else if (index == WGD->totalTimeIncrements - 1) {
      endtime = simTimeStart + WID->fires->fireDur;
    } else {
      endtime = WGD->timestamp[index+1];
    }

    /**
       * Fire time loop for current sensor time
       **/

    while (simTimeCurr < endtime) {
          /**
       * Reset icellflag values
       **/
      WGD->resetICellFlag();

      /**
       * Create initial velocity field from the new sensors
       **/
      WGD->applyWindProfile(WID, index, arguments.solveType);
    
      /**
       * Run ROS model to get initial spread rate and fire properties
       **/

      fire->LevelSetNB(WGD);
    
      /**
       * Calculate fire-induced winds from burning cells
       **/
      fire->potential(WGD);

       /** 
       * Apply parameterizations
       **/
      WGD->applyParametrizations(WID);

      /**
       * Run run wind solver to calculate mass conserved velocity field including fire-induced winds
       **/
      solver->solve(WID, WGD, !arguments.solveWind);
      if (TGD != nullptr) {
	      TGD->run();
      }

      /**
       * Run ROS model to calculate spread rates with updated winds
       **/
      fire->LevelSetNB(WGD);

      /**
       * Advance fire front through level set method
       **/
      fire->move(WGD);

      /**
	     * Advance fire time from variable fire timestep
	     **/
      simTimeCurr += fire->dt;
      
      std::cout << "time = " << simTimeCurr << endl;

      if (plume != nullptr){
	std::cout << "------Running Plume------" << std::endl;
        // Loop through domain to find new smoke sources
	for (int j=1;j<WGD->ny-2;j++){
	  for (int i=1;i<WGD->nx-2;i++){
	    int idx = i+j*(WGD->nx-1);
	    // If smoke flag set in fire program, get x, y, z location and set source
	    if (fire->smoke_flag[idx] == 1){
	      float x_pos = i*WGD->dx;
	      float y_pos = j*WGD->dy;
	      float z_pos = WGD->terrain[idx]+1;
	      float ppt = 20;
	      std::cout<<"x = "<<x_pos<<", y = "<<y_pos<<", z = "<<z_pos<<std::endl;
	      SourceFire *source = new SourceFire(x_pos, y_pos, z_pos, ppt);
	      source->setSource(); 
	      std::vector<Source *> sourceList;
	      sourceList.push_back(dynamic_cast<Source*>(source));
	      // Add source to plume
	      plume->addSources(sourceList);
	      // Clear smoke flag in fire program so no duplicate source set next time step
	      fire->smoke_flag[idx] = 0;
	    }
	  }	  
	}
	std::cout<<"Plume run"<<std::endl;
	QEStime pendtime;///< End time for fire time loop
	pendtime = simTimeCurr; //run until end of fire timestep
	plume->run(pendtime, WGD, TGD, PoutputVec);
	std::cout << "------Plume Finished------" << std::endl;
      }


      /**
	* Save fire data to netCDF file
	**/
      for (auto outItr = outFire.begin(); outItr != outFire.end(); ++outItr) {
        (*outItr)->save(simTimeCurr);
      }

      /**
	 * Reset wind fields to initial values for sensor timestep
	 **/
      //WGD->u0 = Fu0;
      //WGD->v0 = Fv0;
      //WGD->w0 = Fw0;
    }
  

  }
  std::cout << "Simulation finished" << std::endl;
  exit(EXIT_SUCCESS);
}
