#include <iostream>
#include <cmath>
#include <vector>
#include <string>

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

#include "winds/WINDSOutputWRF.h"

#include "winds/TURBGeneralData.h"
#include "winds/TURBOutput.h"

#include "winds/Solver.h"
#include "winds/CPUSolver.h"
#include "winds/DynamicParallelism.h"
#include "winds/GlobalMemory.h"
#include "winds/SharedMemory.h"

#include "winds/Sensor.h"

#include "winds/TextTable.h"


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

// These functions run the solvers and return the solved WINDSGeneralData
WINDSGeneralData *runSerial(WINDSGeneralData *WGD, WINDSInputData *WID, Solver *solverCPU, bool solveWind);
WINDSGeneralData *runDynamic(WINDSGeneralData *WGD_DYNAMIC, WINDSInputData *WID, Solver *solverDynamic, bool solveWind);
WINDSGeneralData *runGlobal(WINDSGeneralData *WGD_GLOBAL, WINDSInputData *WID, Solver *solverGlobal, bool solveWind);
WINDSGeneralData *runShared(WINDSGeneralData *WGD_SHARED, WINDSInputData *WID, Solver *solverShared, bool solveWind);

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
  //WINDSInputData* WID = parseXMLTree(arguments.qesFile);
  WINDSInputData *WID = new WINDSInputData(arguments.qesFile);
  if (!WID) {
    std::cerr << "[ERROR] QUIC Input file: " << arguments.qesFile << " not able to be read successfully." << std::endl;
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

  //
  //Generate general winds data for each solve type.
  //
  WINDSGeneralData *WGD = new WINDSGeneralData(WID, CPU_Type);
  WINDSGeneralData *WGD_DYNAMIC = new WINDSGeneralData(WID, DYNAMIC_P);
  WINDSGeneralData *WGD_GLOBAL = new WINDSGeneralData(WID, Global_M);
  WINDSGeneralData *WGD_SHARED = new WINDSGeneralData(WID, Shared_M);

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
  // Run the QES-Winds Solver
  // //////////////////////////////////////////

  Solver *solverCPU = nullptr, *solverDynamic = nullptr, *solverGlobal = nullptr, *solverShared = nullptr;
  std::vector<WINDSGeneralData *> completedSolvers;
  std::vector<string> solverNames;

  switch (arguments.solveType) {
  case 1:
    WGD = runSerial(WGD, WID, solverCPU, arguments.solveWind);
    completedSolvers.push_back(runDynamic(WGD_DYNAMIC, WID, solverDynamic, arguments.solveWind));
    solverNames.push_back("Dynamic");
    completedSolvers.push_back(runGlobal(WGD_GLOBAL, WID, solverGlobal, arguments.solveWind));
    solverNames.push_back("Global");
    completedSolvers.push_back(runShared(WGD_SHARED, WID, solverShared, arguments.solveWind));
    solverNames.push_back("Shared");
    break;
  case 2:
    WGD = runSerial(WGD, WID, solverCPU, arguments.solveWind);
    completedSolvers.push_back(runDynamic(WGD_DYNAMIC, WID, solverDynamic, arguments.solveWind));
    solverNames.push_back("Dynamic");
    break;
  case 3:
    WGD = runSerial(WGD, WID, solverCPU, arguments.solveWind);
    completedSolvers.push_back(runGlobal(WGD_GLOBAL, WID, solverGlobal, arguments.solveWind));
    solverNames.push_back("Global");
    break;
  case 4:
    WGD = runSerial(WGD, WID, solverCPU, arguments.solveWind);
    completedSolvers.push_back(runShared(WGD_SHARED, WID, solverShared, arguments.solveWind));
    solverNames.push_back("Shared");
    break;
  }

  // Table title
  std::cout << "Performing comparative analysis against CPU serial solver...\n"
            << std::endl;
  TextTable table1('-', '|', '+');
  TextTable table2('-', '|', '+');
  TextTable table3('-', '|', '+');
  // Specify table 1 header row here
  table1.add("SOLVER NAME");
  table1.add("MAX U DIFF");
  table1.add("MAX V DIFF");
  table1.add("MAX W DIFF");
  table1.endOfRow();
  // Specify table 2 header row here
  table2.add("SOLVER NAME");
  table2.add("AVG U DIFF");
  table2.add("AVG V DIFF");
  table2.add("AVG W DIFF");
  table2.endOfRow();
  // Specify table 3 header row here
  table3.add("SOLVER NAME");
  table3.add("MAX WVM DIFF");
  table3.add("AVG WVM DIFF");
  table3.add("R^2 VALUE");
  table3.endOfRow();

  // Loop to calculate comparison metrics between serial and parallel solvers
  for (size_t solversIndex = 0; solversIndex < completedSolvers.size(); ++solversIndex) {
    // Calculating u differences
    float maxUDif = 0;
    float avgUDif = 0;
    float totalUDif = 0;
    float uDif = 0;
    for (size_t uDifIndex = 0; uDifIndex < WGD->u.size(); uDifIndex++) {
      uDif = std::abs(WGD->u[uDifIndex] - completedSolvers[solversIndex]->u[uDifIndex]);
      if (uDif > maxUDif) maxUDif = uDif;
      totalUDif += uDif;
    }
    avgUDif = totalUDif / WGD->u.size();

    // Calculating v differences
    float maxVDif = 0;
    float avgVDif = 0;
    float totalVDif = 0;
    float vDif = 0;
    for (size_t vDifIndex = 0; vDifIndex < WGD->v.size(); vDifIndex++) {
      vDif = std::abs(WGD->v[vDifIndex] - completedSolvers[solversIndex]->v[vDifIndex]);
      if (vDif > maxVDif) maxVDif = vDif;
      totalVDif += vDif;
    }
    avgVDif = totalVDif / WGD->v.size();

    // Calculating w differences
    float maxWDif = 0;
    float avgWDif = 0;
    float totalWDif = 0;
    float wDif = 0;
    for (size_t wDifIndex = 0; wDifIndex < WGD->w.size(); wDifIndex++) {
      wDif = std::abs(WGD->w[wDifIndex] - completedSolvers[solversIndex]->w[wDifIndex]);
      if (wDif > maxWDif) maxWDif = wDif;
      totalWDif += wDif;
    }
    avgWDif = totalWDif / WGD->w.size();

    // Wind velocity magnitude (WindVelMag) difference calculations
    float cpuWVM = 0;
    float gpuWVM = 0;
    float totalWvmDif = 0;
    float avgWvmDif = 0;
    float maxWvmDif = 0;
    float cpuSum = 0;
    float gpuSum = 0;
    // Calculate vector magnitudes, find difference and then add to sum
    for (size_t i = 0; i < WGD->w.size(); ++i) {
      cpuWVM = sqrt(((WGD->u[i]) * (WGD->u[i])) + ((WGD->v[i]) * (WGD->v[i])) + ((WGD->w[i]) * (WGD->w[i])));
      gpuWVM = sqrt(((completedSolvers[solversIndex]->u[i]) * (completedSolvers[solversIndex]->u[i])) + ((completedSolvers[solversIndex]->v[i]) * (completedSolvers[solversIndex]->v[i])) + ((completedSolvers[solversIndex]->w[i]) * (completedSolvers[solversIndex]->w[i])));
      if (std::abs(cpuWVM - gpuWVM) > maxWvmDif) maxWvmDif = std::abs(cpuWVM - gpuWVM);
      totalWvmDif += std::abs(cpuWVM - gpuWVM);
      // These sums required for R-squared calculations below
      cpuSum += cpuWVM;
      gpuSum += gpuWVM;
    }
    avgWvmDif = totalWvmDif / WGD->w.size();

    // /////////////////////////
    // CALCULATING R-squared
    // /////////////////////////

    // R-squared is calculated by first finding r (shocking), and squaring it.
    // To find r, you find the standard deviation of the two data sets,
    // then find the covariance between them. Divide the covariance by the
    // product of the two standard deviations, and you'll find R.
    float cpuAverage = cpuSum / WGD->w.size();
    float gpuAverage = gpuSum / completedSolvers[solversIndex]->w.size();
    float cpuDevSum = 0;
    float gpuDevSum = 0;
    float solversCovariance = 0;
    float solversCovarianceSum = 0;
    for (size_t i = 0; i < WGD->w.size(); i++) {
      // Calculate vector magnitudes
      cpuWVM = sqrt(((WGD->u[i]) * (WGD->u[i])) + ((WGD->v[i]) * (WGD->v[i])) + ((WGD->w[i]) * (WGD->w[i])));
      gpuWVM = sqrt(((completedSolvers[solversIndex]->u[i]) * (completedSolvers[solversIndex]->u[i])) + ((completedSolvers[solversIndex]->v[i]) * (completedSolvers[solversIndex]->v[i])) + ((completedSolvers[solversIndex]->w[i]) * (completedSolvers[solversIndex]->w[i])));
      // Calculate deviation and covariance sums
      cpuDevSum += (cpuWVM - cpuAverage) * (cpuWVM - cpuAverage);
      gpuDevSum += (gpuWVM - gpuAverage) * (gpuWVM - gpuAverage);
      solversCovarianceSum += (cpuWVM - cpuAverage) * (gpuWVM - gpuAverage);
    }
    // Calculate standard deviations and covariance
    float cpuStDev = std::sqrt(cpuDevSum / (WGD->w.size() - 1));
    float gpuStDev = std::sqrt(gpuDevSum / (WGD->w.size() - 1));
    solversCovariance = solversCovarianceSum / (WGD->w.size() - 1);
    // Calculate r and r^2
    float r = solversCovariance / (cpuStDev * gpuStDev);
    float rSquared = r * r;

    // Table comparison metrics row
    table1.add(solverNames[solversIndex]);
    table1.add(std::to_string(maxUDif));
    table1.add(std::to_string(maxVDif));
    table1.add(std::to_string(maxWDif));
    table1.endOfRow();
    table2.add(solverNames[solversIndex]);
    table2.add(std::to_string(avgUDif));
    table2.add(std::to_string(avgVDif));
    table2.add(std::to_string(avgWDif));
    table2.endOfRow();
    table3.add(solverNames[solversIndex]);
    table3.add(std::to_string(maxWvmDif));
    table3.add(std::to_string(avgWvmDif));
    table3.add(std::to_string(rSquared));
    table3.endOfRow();

    //std::cout << "  Max u difference: " << maxUDif << std::endl;
    //std::cout << "  Max v difference: " << maxVDif << std::endl;
    //std::cout << "  Max w difference: " << maxWDif << std::endl;
    //std::cout << "  Average u difference: " << avgUDif << std::endl;
    //std::cout << "  Average v difference: " << avgVDif << std::endl;
    //std::cout << "  Average w difference: " << avgWDif << std::endl;
    //std::cout << "  Total u difference: " << totalUDif << std::endl;
    //std::cout << "  Total v difference: " << totalVDif << std::endl;
    //std::cout << "  Total w difference: " << totalWDif << std::endl;
    //std::cout << "  Total WindVelMag difference: " << totalWvmDif << std::endl;
    //std::cout << "  CPU STDEV: " << cpuStDev << std::endl;
    //std::cout << "  GPU STDEV: " << gpuStDev << std::endl;
    //std::cout << "  Covariance: " << solversCovariance << std::endl;
    //std::cout << "  R value: " << r << std::endl;
    //std::cout << "  R-squared value: " << rSquared << std::endl;
    //std::cout << std::endl;
  }
  // Print comparison table
  std::cout << "Table 1 of 3\n";
  std::cout << table1 << std::endl;
  std::cout << "Table 2 of 3\n";
  std::cout << table2 << std::endl;
  std::cout << "Table 3 of 3\n";
  std::cout << table3 << std::endl;
  std::cout << "Comparative analysis complete!\n";

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

//CPU solver
WINDSGeneralData *runSerial(WINDSGeneralData *WGD, WINDSInputData *WID, Solver *solverCPU, bool solveWind)
{
  std::cout << std::endl;
  std::cout << "Run Serial Solver (CPU)..." << std::endl;
  solverCPU = new CPUSolver(WID, WGD);

  // Reset icellflag values
  WGD->resetICellFlag();

  // Create initial velocity field from the new sensors
  WGD->applyWindProfile(WID, 0, CPU_Type);

  // Apply parametrizations
  WGD->applyParametrizations(WID);

  solverCPU->solve(WID, WGD, !solveWind);
  std::cout << "CPU solver done!\n";
  std::cout << std::endl;
  return WGD;
}

//Dynamic parallel GPU solver
WINDSGeneralData *runDynamic(WINDSGeneralData *WGD_DYNAMIC, WINDSInputData *WID, Solver *solverDynamic, bool solveWind)
{
  std::cout << "Run Dynamic Parallel Solver (GPU)..." << std::endl;
  solverDynamic = new DynamicParallelism(WID, WGD_DYNAMIC);

  // Reset icellflag values
  WGD_DYNAMIC->resetICellFlag();

  // Create initial velocity field from the new sensors
  WGD_DYNAMIC->applyWindProfile(WID, 0, DYNAMIC_P);

  // Apply parametrizations
  WGD_DYNAMIC->applyParametrizations(WID);

  solverDynamic->solve(WID, WGD_DYNAMIC, !solveWind);
  std::cout << "Dynamic solver done!\n";
  std::cout << std::endl;
  return WGD_DYNAMIC;
}

//Global memory GPU solver
WINDSGeneralData *runGlobal(WINDSGeneralData *WGD_GLOBAL, WINDSInputData *WID, Solver *solverGlobal, bool solveWind)
{
  std::cout << "Run Global Memory Solver (GPU)..." << std::endl;
  solverGlobal = new GlobalMemory(WID, WGD_GLOBAL);

  // Reset icellflag values
  WGD_GLOBAL->resetICellFlag();

  // Create initial velocity field from the new sensors
  WGD_GLOBAL->applyWindProfile(WID, 0, Global_M);

  // Apply parametrizations
  WGD_GLOBAL->applyParametrizations(WID);

  solverGlobal->solve(WID, WGD_GLOBAL, !solveWind);
  std::cout << "Global solver done!\n";
  std::cout << std::endl;
  return WGD_GLOBAL;
}

//Shared memory GPU solver
WINDSGeneralData *runShared(WINDSGeneralData *WGD_SHARED, WINDSInputData *WID, Solver *solverShared, bool solveWind)
{
  std::cout << "Run Shared Memory Solver (GPU)..." << std::endl;
  solverShared = new SharedMemory(WID, WGD_SHARED);

  // Reset icellflag values
  WGD_SHARED->resetICellFlag();

  // Create initial velocity field from the new sensors
  WGD_SHARED->applyWindProfile(WID, 0, Shared_M);

  // Apply parametrizations
  WGD_SHARED->applyParametrizations(WID);

  solverShared->solve(WID, WGD_SHARED, !solveWind);
  std::cout << "Shared solver done!\n";
  std::cout << std::endl;
  return WGD_SHARED;
}
