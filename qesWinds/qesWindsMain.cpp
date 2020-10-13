#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

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

namespace pt = boost::property_tree;

/**
 * This function takes in a filename and attempts to open and parse it.
 * If the file can't be opened or parsed properly it throws an exception,
 * if the file is missing necessary data, an error will be thrown detailing
 * what data and where in the xml the data is missing. If the tree can't be
 * parsed, the Root* value returned is 0, which will register as false if tested.
 * @param fileName the path/name of the file to be opened, must be an xml
 * @return A pointer to a root that is filled with data parsed from the tree
 */
WINDSInputData* parseXMLTree(const std::string fileName);
Sensor* parseSensors (const std::string fileName);

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
    WINDSInputData* WID = parseXMLTree(arguments.quicFile);
    if ( !WID ) {
        std::cerr << "[ERROR] QUIC Input file: " << arguments.quicFile <<
            " not able to be read successfully." << std::endl;
        exit(EXIT_FAILURE);
    }

    // If the sensor file specified in the xml
    if (WID->metParams->sensorName.size() > 0)
    {
        for (auto i = 0; i < WID->metParams->sensorName.size(); i++)
  		  {
            WID->metParams->sensors.push_back(new Sensor());            // Create new sensor object
            WID->metParams->sensors[i] = parseSensors(WID->metParams->sensorName[i]);       // Parse new sensor objects from xml
        }
    }


    // Checking if
    if (arguments.compTurb && !WID->localMixingParam) {
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
    WINDSGeneralData* WGD = new WINDSGeneralData(WID);

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
        TGD = new TURBGeneralData(WGD);
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
        TGD->run(WGD);
    }

    // /////////////////////////////
    // Output the various files requested from the simulation run
    // (netcdf wind velocity, icell values, etc...
    // /////////////////////////////
    for(auto id_out=0u;id_out<outputVec.size();id_out++)
    {
        outputVec.at(id_out)->save(0.0); // need to replace 0.0 with timestep
    }

    if (WID->simParams->totalTimeIncrements > 1)
    {
      for (int index = 1; index < WID->simParams->totalTimeIncrements; index++)
      {
        // Reset icellflag values
        for (int k = 0; k < WGD->nz-2; k++)
        {
            for (int j = 0; j < WGD->ny-1; j++)
            {
                for (int i = 0; i < WGD->nx-1; i++)
                {
                    int icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
                    if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2 && WGD->icellflag[icell_cent] != 8 && WGD->icellflag[icell_cent] != 7)
                    {
                      WGD->icellflag[icell_cent] = 1;
                    }
                }
            }
        }

        // Create initial velocity field from the new sensors
        WID->metParams->sensors[0]->inputWindProfile(WID, WGD, index);

        // ///////////////////////////////////////
        // Canopy Vegetation Parameterization
        // ///////////////////////////////////////
        for (size_t i = 0; i < WGD->allBuildingsV.size(); i++)
        {
            // for now this does the canopy stuff for us
            WGD->allBuildingsV[WGD->building_id[i]]->canopyVegetation(WGD);
        }

        ///////////////////////////////////////////
        //   Upwind Cavity Parameterization     ///
        ///////////////////////////////////////////
        if (WID->simParams->upwindCavityFlag > 0)
        {
            std::cout << "Applying upwind cavity parameterization...\n";
            for (size_t i = 0; i < WGD->allBuildingsV.size(); i++)
            {
                WGD->allBuildingsV[WGD->building_id[i]]->upwindCavity(WID, WGD);
            }
            std::cout << "Upwind cavity parameterization done...\n";
        }

        //////////////////////////////////////////////////
        //   Far-Wake and Cavity Parameterizations     ///
        //////////////////////////////////////////////////
        if (WID->simParams->wakeFlag > 0)
        {
            std::cout << "Applying wake behind building parameterization...\n";
            for (size_t i = 0; i < WGD->allBuildingsV.size(); i++)
            {
                WGD->allBuildingsV[WGD->building_id[i]]->polygonWake(WID, WGD, WGD->building_id[i]);
            }
            std::cout << "Wake behind building parameterization done...\n";
        }

        ///////////////////////////////////////////
        //   Street Canyon Parameterization     ///
        ///////////////////////////////////////////
        if (WID->simParams->streetCanyonFlag > 0)
        {
            std::cout << "Applying street canyon parameterization...\n";
            for (size_t i = 0; i < WGD->allBuildingsV.size(); i++)
            {
                WGD->allBuildingsV[WGD->building_id[i]]->streetCanyon(WGD);
            }
            std::cout << "Street canyon parameterization done...\n";
        }

        ///////////////////////////////////////////
        //      Sidewall Parameterization       ///
        ///////////////////////////////////////////
        if (WID->simParams->sidewallFlag > 0)
        {
            std::cout << "Applying sidewall parameterization...\n";
            for (size_t i = 0; i < WGD->allBuildingsV.size(); i++)
            {
                WGD->allBuildingsV[WGD->building_id[i]]->sideWall(WID, WGD);
            }
            std::cout << "Sidewall parameterization done...\n";
        }


        ///////////////////////////////////////////
        //      Rooftop Parameterization        ///
        ///////////////////////////////////////////
        if (WID->simParams->rooftopFlag > 0)
        {
            std::cout << "Applying rooftop parameterization...\n";
            for (size_t i = 0; i < WGD->allBuildingsV.size(); i++)
            {
                WGD->allBuildingsV[WGD->building_id[i]]->rooftop (WID, WGD);
            }
            std::cout << "Rooftop parameterization done...\n";
        }

        WGD->wall->setVelocityZero (WGD);

        // Run WINDS simulation code
        solver->solve(WID, WGD, !arguments.solveWind );

        std::cout << "Solver done!\n";

        // /////////////////////////////
        // Output the various files requested from the simulation run
        // (netcdf wind velocity, icell values, etc...
        // /////////////////////////////
        for(auto id_out=0u;id_out<outputVec.size();id_out++)
        {
            outputVec.at(id_out)->save((float) index);
        }

      }

    }


    // /////////////////////////////
    exit(EXIT_SUCCESS);
}

WINDSInputData* parseXMLTree(const std::string fileName)
{
	pt::ptree tree;

	try
	{
		pt::read_xml(fileName, tree);
	}
	catch (boost::property_tree::xml_parser::xml_parser_error& e)
	{
		std::cerr << "Error reading tree in" << fileName << "\n";
		return (WINDSInputData*)0;
	}

	WINDSInputData* xmlRoot = new WINDSInputData();
        xmlRoot->parseTree( tree );
	return xmlRoot;
}


Sensor* parseSensors (const std::string fileName)
{

  pt::ptree tree1;

  try
  {
    pt::read_xml(fileName, tree1);
  }
  catch (boost::property_tree::xml_parser::xml_parser_error& e)
  {
    std::cerr << "Error reading tree in" << fileName << "\n";
    return (Sensor*)0;
  }

  Sensor* xmlRoot = new Sensor();
  xmlRoot->parseTree( tree1 );
  return xmlRoot;

}
