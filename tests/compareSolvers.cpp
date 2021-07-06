#include <iostream>
#include <cmath>

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

  /*
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
  */

  
  //
  //Generate general winds data for each solve type.
  //
  WINDSGeneralData *WGD = new WINDSGeneralData(WID, CPU_Type);
  WINDSGeneralData *WGD_DYNAMIC = new WINDSGeneralData(WID, DYNAMIC_P);
  // WINDSGeneralData *WGD_GLOBAL = new WINDSGeneralData(WID, Global_M);
  // WINDSGeneralData *WGD_SHARED = new WINDSGeneralData(WID, Shared_M); 

  /*
  //OUTPUTS NOT NEEDED FOR THIS TEST
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
  */


  //
  //Turbulence not accounted for in this test
  //
  // Generate the general TURB data from WINDS data
  // based on if the turbulence output file is defined
  //TURBGeneralData *TGD = nullptr;
  //(void) TGD;
  /*
  if (arguments.compTurb) {
    TGD = new TURBGeneralData(WID, WGD);
  }
  if (arguments.compTurb && arguments.turbOutput) {
    outputVec.push_back(new TURBOutput(TGD, arguments.netCDFFileTurb));
  }
  */

  std::cout << "Running time step: " << to_iso_extended_string(WGD->timestamp[0]) << std::endl;

  // //////////////////////////////////////////
  //
  // Run the QES-Winds Solver
  //
  // //////////////////////////////////////////

  Solver *solverCPU, *solverDynamic, *solverGlobal, *solverShared = nullptr;
  std::cout << "Run Serial Solver (CPU) ..." << std::endl;
  solverCPU = new CPUSolver(WID, WGD);
  std::cout << "Run Dynamic Parallel Solver (GPU) ..." << std::endl;
  solverDynamic = new DynamicParallelism(WID, WGD_DYNAMIC);
  //std::cout << "Run Global Memory Solver (GPU) ..." << std::endl;
  //solverGlobal = new GlobalMemory(WID, WGD_GLOBAL);
  //std::cout << "Run Shared Memory Solver (GPU) ..." << std::endl;
  //solverShared = new SharedMemory(WID, WGD_SHARED);

  
  solverDynamic->solve(WID, WGD_DYNAMIC, !arguments.solveWind); 
  solverCPU->solve(WID, WGD, !arguments.solveWind);
  //solverGlobal->solve(WID, WGD_GLOBAL, !arguments.solveWind);
  //solverShared->solve(WID, WGD_SHARED, !arguments.solveWind);

  std::cout << "Solvers done!\n";
  
  ////
  //Calculating absoulte differences beetween CPU and dynamic parallel solvers.
  ////

  //calculating u differences
  float maxUDif = 0;
  float avgUDif;
  float totalUDif = 0;
  float uDif = 0;
  for(size_t uDifIndex = 0; uDifIndex < WGD->u.size(); uDifIndex++){
    uDif = std::abs(WGD->u[uDifIndex] - WGD_DYNAMIC->u[uDifIndex]);
    if(uDif>maxUDif) maxUDif = uDif;
    totalUDif += std::abs((WGD->u[uDifIndex] - WGD_DYNAMIC->u[uDifIndex]));
    //percentDif = (WGD->u[difIndex] - WGD_DYNAMIC->u[difIndex]) / WGD->u[difIndex] * 100;
    //std::cout << "CPU u values: " << WGD->u[difIndex] << std::endl;
    //std::cout << "GPU u values: " << WGD_DYNAMIC->u[difIndex] << std::endl;
  }
  avgUDif = totalUDif/WGD->u.size();

  //calculating v differences
  float maxVDif = 0;
  float avgVDif;
  float totalVDif = 0;
  float vDif = 0;
  for(size_t vDifIndex = 0; vDifIndex < WGD->v.size(); vDifIndex++){
    vDif = std::abs(WGD->v[vDifIndex] - WGD_DYNAMIC->v[vDifIndex]);
    if(vDif>maxVDif) maxVDif = vDif;
    totalVDif += std::abs((WGD->v[vDifIndex] - WGD_DYNAMIC->v[vDifIndex]));
  }
  avgVDif = totalVDif/WGD->v.size();

  //calculating w differences
  float maxWDif = 0;
  float avgWDif;
  float totalWDif = 0;
  float wDif = 0;
  for(size_t wDifIndex = 0; wDifIndex < WGD->w.size(); wDifIndex++){
    wDif = std::abs(WGD->w[wDifIndex] - WGD_DYNAMIC->w[wDifIndex]);
    if(wDif>maxWDif) maxWDif = wDif;
    totalWDif += std::abs((WGD->w[wDifIndex] - WGD_DYNAMIC->w[wDifIndex]));
  }
  avgWDif = totalWDif/WGD->w.size();

  //displaying max differences
  std::cout << "Max u difference: " << maxUDif << std::endl;
  std::cout << "Max v difference: " << maxVDif << std::endl;
  std::cout << "Max w difference: " << maxWDif << std::endl;
  //displaying average differences
  std::cout << "Average u difference: " << avgUDif << std::endl;
  std::cout << "Average v difference: " << avgVDif << std::endl;
  std::cout << "Average w difference: " << avgWDif << std::endl;
  //displaying sum of difference
  std::cout << "Total u difference: " << totalUDif << std::endl;
  std::cout << "Total v difference: " << totalVDif << std::endl;
  std::cout << "Total w difference: " << totalWDif << std::endl;

  //you could compute the windVelMag at each cell

  // average u difference =
  // average v difference =
  // average w difference =
  // maybe the max differences...

  // even better would be to compute a R^2 regression comparison...

  //if (TGD != nullptr)
  //  TGD->run(WGD);

  // /////////////////////////////
  // Output the various files requested from the simulation run
  // (netcdf wind velocity, icell values, etc...
  // /////////////////////////////
  //for (auto id_out = 0u; id_out < outputVec.size(); id_out++) {
  //  outputVec.at(id_out)->save(WGD->timestamp[0]);
  //}

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
  /*
  for (int index = 1; index < WID->simParams->totalTimeIncrements; index++) {
    std::cout << "Running time step: " << to_iso_extended_string(WGD_DYNAMIC->timestamp[index]) << std::endl;
    // Reset icellflag values
    WGD_DYNAMIC->resetICellFlag();

    // Create initial velocity field from the new sensors
    WID->metParams->sensors[0]->inputWindProfile(WID, WGD_DYNAMIC, index, DYNAMIC_P);

    // Apply parametrizations
    WGD_DYNAMIC->applyParametrizations(WID);

    // Run WINDS simulation code
    solverDynamic->solve(WID, WGD_DYNAMIC, !arguments.solveWind);

    std::cout << "Solver done!\n";

    //if (TGD != nullptr)
    //  TGD->run(WGD);

    // /////////////////////////////
    // Output the various files requested from the simulation run
    // (netcdf wind velocity, icell values, etc...
    // /////////////////////////////
    //for (auto id_out = 0u; id_out < outputVec.size(); id_out++) {
    //  outputVec.at(id_out)->save(WGD->timestamp[index]);
    //}
  }
  */

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
