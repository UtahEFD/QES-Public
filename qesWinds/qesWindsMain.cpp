#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "util/QESNetCDFOutput.h"

#include "src/winds/handleWINDSArgs.h"

#include "src/winds/WINDSInputData.h"
#include "src/winds/WINDSGeneralData.h"
#include "src/winds/WINDSOutputVisualization.h"
#include "src/winds/WINDSOutputWorkspace.h"

#include "src/winds/WINDSOutputWRF.h"

#include "src/winds/TURBGeneralData.h"
#include "src/winds/TURBOutput.h"

#include "src/winds/Solver.h"
#include "src/winds/CPUSolver.h"
#include "src/winds/DynamicParallelism.h"
#include "src/winds/GlobalMemory.h"
#include "src/winds/SharedMemory.h"

#include "src/winds/Sensor.h"

namespace pt = boost::property_tree;

using namespace boost::gregorian;
using namespace boost::posix_time;

/**
 * This function takes in a filename and attempts to open and parse it.
 * If the file can't be opened or parsed properly it throws an exception,
 * if the file is missing necessary data, an error will be thrown detailing
 * what data and where in the xml the data is missing. If the tree can't be
 * parsed, the Root* value returned is 0, which will register as false if tested.
 * @param fileName the path/name of the file to be opened, must be an xml
 * @return A pointer to a root that is filled with data parsed from the tree
 */
WINDSInputData *parseXMLTree(const std::string fileName);
Sensor *parseSensors(const std::string fileName);

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
  WINDSInputData *WID = new WINDSInputData(arguments.quicFile);
  if (!WID) {
    std::cerr << "[ERROR] QUIC Input file: " << arguments.quicFile << " not able to be read successfully." << std::endl;
    exit(EXIT_FAILURE);
  }


  /*
    // If the sensor file specified in the xml
    if (WID->metParams->sensorName.size() > 0)
    {
        for (auto i = 0; i < WID->metParams->sensorName.size(); i++)
  		  {
            WID->metParams->sensors.push_back(new Sensor());            // Create new sensor object
            WID->metParams->sensors[i] = parseSensors(WID->metParams->sensorName[i]);       // Parse new sensor objects from xml
        }
    }
    */

  // Checking if
  if (arguments.compTurb && !WID->turbParams) {
    std::cerr << "[ERROR] Turbulence model is turned on without turbParams in QES Intput file "
              << arguments.quicFile << std::endl;
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

  std::cout << "Running time step: " << to_iso_extended_string(WGD->timestamp[0]) << std::endl;

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

  solver->solve(WID, WGD, !arguments.solveWind);

  std::cout << "Solver done!\n";

  if (TGD != nullptr)
    TGD->run(WGD);

  // /////////////////////////////
  // Output the various files requested from the simulation run
  // (netcdf wind velocity, icell values, etc...
  // /////////////////////////////
  for (auto id_out = 0u; id_out < outputVec.size(); id_out++) {
    outputVec.at(id_out)->save(WGD->timestamp[0]);
  }

  /*
    for(size_t index=0; index < WGD->timestamp.size(); index++)
        std::cout << to_iso_extended_string(WGD->timestamp[index]) << std::endl;

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


    if (WID->simParams->wrfInputData) {

        // WID->simParams->outputWRFData();
    }


    // /////////////////////////////
    // Output the various files requested from the simulation run
    // (netcdf wind velocity, icell values, etc...
    // /////////////////////////////
    for(auto id_out=0u;id_out<outputVec.size();id_out++)
    {
        outputVec.at(id_out)->save(0.0); // need to replace 0.0 with timestep
    }
    */

  for (int index = 1; index < WID->simParams->totalTimeIncrements; index++) {
    std::cout << "Running time step: " << to_iso_extended_string(WGD->timestamp[index]) << std::endl;
    // Reset icellflag values
    WGD->resetICellFlag();

    // Create initial velocity field from the new sensors
    WID->metParams->sensors[0]->inputWindProfile(WID, WGD, index, arguments.solveType);

    // Apply parametrizations
    WGD->applyParametrizations(WID);

    // Run WINDS simulation code
    solver->solve(WID, WGD, !arguments.solveWind);

    std::cout << "Solver done!\n";

    if (TGD != nullptr)
      TGD->run(WGD);

    // /////////////////////////////
    // Output the various files requested from the simulation run
    // (netcdf wind velocity, icell values, etc...
    // /////////////////////////////
    for (auto id_out = 0u; id_out < outputVec.size(); id_out++) {
      outputVec.at(id_out)->save(WGD->timestamp[index]);
    }
  }

  // /////////////////////////////
  exit(EXIT_SUCCESS);
}

WINDSInputData *parseXMLTree(const std::string fileName)
{
  pt::ptree tree;

  try {
    pt::read_xml(fileName, tree);
  } catch (boost::property_tree::xml_parser::xml_parser_error &e) {
    std::cerr << "Error reading tree in" << fileName << "\n";
    return (WINDSInputData *)0;
  }

  WINDSInputData *xmlRoot = new WINDSInputData();
  xmlRoot->parseTree(tree);
  return xmlRoot;
}


Sensor *parseSensors(const std::string fileName)
{

  pt::ptree tree1;

  try {
    pt::read_xml(fileName, tree1);
  } catch (boost::property_tree::xml_parser::xml_parser_error &e) {
    std::cerr << "Error reading tree in" << fileName << "\n";
    return (Sensor *)0;
  }

  Sensor *xmlRoot = new Sensor();
  xmlRoot->parseTree(tree1);
  return xmlRoot;
}
