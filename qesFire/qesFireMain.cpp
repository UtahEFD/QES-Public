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
 * Copyright (c) 2022 Matthew Moody
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

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "util/ParseException.h"
#include "util/ParseInterface.h"
#include "util/QEStool.h"
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

#include "fire/Fire.h"
#include "fire/FIREOutput.h"
#include <chrono>
namespace pt = boost::property_tree;

using namespace boost::gregorian;
using namespace boost::posix_time;


int main(int argc, char *argv[])
{
    // QES-Winds - Version output information
    std::string Revision = "0";
    std::cout << "QES-Fire " << "1.0.0" << std::endl;

#ifdef HAS_OPTIX
    std::cout << "OptiX is available!" << std::endl;
#endif

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
    QEStool::error("QES Input file: " + arguments.qesWindsParamFile + " not able to be read successfully.");
  }

  // Checking if
  if (arguments.compTurb && !WID->turbParams) {
    QEStool::error("Turbulence model is turned on without turbParams in QES Intput file "
                   + arguments.qesWindsParamFile);
  }


  if (arguments.terrainOut) {
    if (WID->simParams->DTE_heightField) {
      std::cout << "Creating terrain OBJ....\n";
      WID->simParams->DTE_heightField->outputOBJ(arguments.filenameTerrain);
      std::cout << "OBJ created....\n";
    } else {
      QEStool::error("No dem file specified as input");
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
    TURBGeneralData* TGD = nullptr;
    if (arguments.compTurb) {
        TGD = new TURBGeneralData(WID,WGD);
    }
    if (arguments.compTurb && arguments.turbOutput) {
        outputVec.push_back(new TURBOutput(TGD,arguments.netCDFFileTurb));
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
    QEStool::error("Invalid solve type");
  }




    // /////////////////////////////
    //
    // Run Fire Code
    //
    // ///////////////////////////// 
 
    /** 
     * Create Fire Map
     **/

    Fire* fire = new Fire(WID, WGD);
 
    /**
     * Create FIREOutput manager
     **/
    
    std::vector<QESNetCDFOutput *> outFire;
    std::cout << "test" << std::endl;
    outFire.push_back(new FIREOutput(WGD, fire, arguments.netCDFFileFireOut));


    /**
     * Time variables to track fire time and sensor timesteps
     **/ 
    QEStime simTimeStart = WGD->timestamp[0];
    QEStime simTimeCurr = simTimeStart;

  
    std::vector<float> Fu0; //
    std::vector<float> Fv0;
    std::vector<float> Fw0;

	/**
	 * Loop  for sensor time data
	 **/
	
    for (int index = 0; index < WGD->totalTimeIncrements; index++) {
      std::cout << "----------------------------------------" << std::endl;
      std::cout << "New Sensor Data" << std::endl;
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
      solver->solve(WID, WGD, !arguments.solveWind );

      std::cout << "Solver done!\n";

      
      /**
       * Run turbulence if specified
       **/
 
      if(TGD != nullptr) {
        TGD->run();
      }
      
      /**
       * Save initial fields from sensor time to reset after each time+fire loop
       **/
      Fu0 = WGD->u0; ///< Initial u-velocity for sensor timestep
      Fv0 = WGD->v0; ///< Initial v-velocity for sensor timestep
      Fw0 = WGD->w0; ///< Initial w-velocity for sensor timestep
      

      simTimeCurr = WGD->timestamp[index]; ///< Simulation time for current sensor time
      QEStime endtime; ///< End time for fire time loop 
      if (WGD->totalTimeIncrements == 1){
	endtime = WGD->timestamp[index] + WID->fires->fireDur;
      } else if (index == WGD->totalTimeIncrements - 1) {
	endtime = WGD->timestamp[index] + (WGD->timestamp[index] - WGD->timestamp[index-1]);
      } else {
        endtime = WGD->timestamp[index+1];
      }

      /**
       * Fire time loop for current sensor time
       **/
      
    while (simTimeCurr < endtime) {
      /**
       * Run ROS model to get initial spread rate and fire properties
       **/
      fire->run(solver, WGD);

      /**
       * Calculate fire-induced winds from burning cells
       **/
      auto start = std::chrono::high_resolution_clock::now(); // Start recording executiontime 
      fire->potential(WGD);
      auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time
	    
      std::chrono::duration<float> elapsed = finish - start;
      std::cout << "Plume solve: elapsed time: " << elapsed.count() << " s\n";   // Print out elapsed execution time for fire-induced winds

      /**
       * Run run wind solver to calculate mass conserved velocity field including fire-induced winds
       **/  
      solver->solve(WID, WGD, !arguments.solveWind);

      /**
       * Run ROS model to calculate spread rates with updated winds
       **/
      fire->run(solver, WGD);

      /**
       * Advance fire front through level set method
       **/
       fire->move(solver, WGD);
        

 
       /**
	* Advance fire time from variable fire timestep
	**/ 
       simTimeCurr += fire->dt;
       std::cout << "time = " << simTimeCurr <<endl;


       /**
	* Save fire data to netCDF file
	**/
	for (auto outItr = outFire.begin(); outItr != outFire.end(); ++outItr){
	  (*outItr)->save(simTimeCurr);
	}

	/**
	 * Reset wind fieldsto initial values for sensor timestep
	 **/
	WGD->u0 = Fu0;
	WGD->v0 = Fv0;
	WGD->w0 = Fw0;
    }        

    }

    
    exit(EXIT_SUCCESS);
}



