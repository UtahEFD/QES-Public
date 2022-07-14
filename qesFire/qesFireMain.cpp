#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

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

#include "fire/Fire.hpp"
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

    // Reset icellflag values
    WGD->resetICellFlag();
    int index = 0;
    // Create initial velocity field from the new sensors
    WGD->applyWindProfile(WID, index, arguments.solveType);
    
    // Apply parametrizations
    WGD->applyParametrizations(WID);

    // Run WINDS simulation code
    solver->solve(WID, WGD, !arguments.solveWind );

    std::cout << "Solver done!\n";

    if (solverC != nullptr) {
        std::cout << "Running comparson type...\n";
        solverC->solve(WID, WGD, !arguments.solveWind);
    }
    // /////////////////////////////
    //
    // Run turbulence
    //
    // /////////////////////////////

    if(TGD != nullptr) {
        TGD->run();
    }

    // /////////////////////////////
    //
    // Run Fire Code
    //
    // ///////////////////////////// 
 
    /** 
     * Create Fire Mapper
     **/
    auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time
    Fire* fire = new Fire(WID, WGD);
    auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time
	    
    std::chrono::duration<float> elapsed = finish - start;
    std::cout << "Fire Map created: elapsed time: " << elapsed.count() << " s\n";   // Print out elapsed execution time

    // create FIREOutput manager
    FIREOutput* fireOutput = new FIREOutput(WGD,fire,arguments.netCDFFileFire);
  
    // set base w in fire model to initial w0
    //fire->w_base = w0;
    
    // Run initial solver to generate full field
    solver->solve(WID, WGD, !arguments.solveWind);

    // save initial fields in solver and fire
    //if (output != nullptr) {
    //    WGD->save();
    //}
    // save initial fields to reset after each time+fire loop
    std::vector<float> u0 = WGD->u0;
    std::vector<float> v0 = WGD->v0;
    std::vector<float> w0 = WGD->w0;

    // save any fire data (at time 0)
    fireOutput->save(0.0);
	
    // Run WINDS simulation code
    std::cout<<"===================="<<std::endl;
    double t = 0;
    
    std::cout<<"total time inc: :"<<WID->simParams->totalTimeIncrements<<std::endl;
    while (t<WID->simParams->totalTimeIncrements) {
        
        std::cout<<"Processing time t = "<<t<<std::endl;
        // re-set initial fields after first time step
        if (t>0) {
		// Reset icellflag values
    		 WGD->resetICellFlag();
	    
	     WGD->u0 = u0;
	     WGD->v0 = v0;
	     WGD->w0 = w0;
		 std::cout<<"Wind field reset to initial"<<std::endl;
	   
	    /*
	    WID->metParams->z0_domain_flag=1;
	    WID->metParams->sensors[0]->inputWindProfile(WID, WGD);
            solver->solve(WID, WGD, !arguments.solveWind);
	    */

	// Generate the general WINDS data from all inputs
    // WINDSGeneralData* WGD = new WINDSGeneralData(WID, arguments.solveType);
	    // Apply parametrizations
    // WGD->applyParametrizations(WID);

    // Run WINDS simulation code
    // solver->solve(WID, WGD, !arguments.solveWind );

    // std::cout << "Solver done!\n";
        }

        // loop 2 times for fire
        int loop = 0;
        while (loop<1) {
            // run Balbi model to get new spread rate and fire properties
            fire->run(solver, WGD);
	    //WGD->applyParametrizations(WID);
            // calculate plume potential
	    auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time
	    
	    fire->potential(WGD);
	    
	    auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time
	    
    	    std::chrono::duration<float> elapsed = finish - start;
    	    std::cout << "Plume solve: elapsed time: " << elapsed.count() << " s\n";   // Print out elapsed execution time
	    /**
            * Apply parameterizations and run wind solver 
	    **/
	     
	    solver->solve(WID, WGD, !arguments.solveWind);
	    // run wind solver
            // solver->solve(WID, WGD, !arguments.solveWind);


            

	    	    
            //increment fire loop
            loop += 1;
   	            
            std::cout<<"--------------------"<<std::endl;
        }
        
        // move the fire
        fire->move(solver, WGD);
        
        // save any fire data
        fireOutput->save(fire->time);
 
        // advance time 
        t = fire->time;
        

    }

    
    exit(EXIT_SUCCESS);
}



