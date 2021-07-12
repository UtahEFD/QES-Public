/* 
  Bar-chart Author: Michael Thomas Greer
  Source: http://cplusplus.com/forum/beginner/264784/
  Date: 11 Jul 2021
*/

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <sstream>

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

#include "TextTable.h"

// For printing comparison results bar chart
// The isatty() function will tell us whether standard input is piped or not.
#ifdef _WIN32
  #include <windows.h>
  #include <io.h>
  #define isatty _isatty
#else
  #include <unistd.h>
#endif

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
WINDSGeneralData* runSerial(WINDSGeneralData* WGD, WINDSInputData* WID, Solver *solverCPU, bool solveWind);
WINDSGeneralData* runDynamic(WINDSGeneralData* WGD_DYNAMIC, WINDSInputData* WID, Solver *solverDynamic, bool solveWind);
WINDSGeneralData* runGlobal(WINDSGeneralData* WGD_GLOBAL, WINDSInputData* WID, Solver *solverGlobal, bool solveWind);
WINDSGeneralData* runShared(WINDSGeneralData* WGD_SHARED, WINDSInputData* WID, Solver *solverShared, bool solveWind);

// This function prints the comparison results into a bar chart
void printChart(string title, bool isHigherBetter, vector<double> results);

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

  Solver *solverCPU, *solverDynamic, *solverGlobal, *solverShared = nullptr;
  std::vector<WINDSGeneralData *> completedSolvers;
  std::vector<string> solverNames;

  switch(arguments.solveType){
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
  std::cout << "Performing comparative analysis against CPU serial solver...\n";
  TextTable t( '-', '|', '+' );
  // Specify table header row here
  t.add( "SOLVER NAME" );
  t.add( "MAX U DIFF" );
  t.add( "MAX V DIFF" );
  t.add( "MAX W DIFF" );
  t.add( "AVG U DIFF" );
  t.add( "AVG V DIFF" );
  t.add( "AVG W DIFF" );
  t.add( "WindVelMag DIFF" );
  t.add( "R^2 VALUE" );
  t.add( "Mean WVM DIFF");
  t.add( "Max WVM DIFF");
  t.endOfRow();

  // These vars for storing comparison results for later bar chart construction in loop below
  vector<double> wvmResults;
  vector<double> rSquaredResults;
  vector<double> avgWVMResults;
  // Loop to calculate comparison metrics between serial and parallel solvers
  for (int solversIndex = 0; solversIndex < completedSolvers.size(); ++solversIndex) {
    // Calculating u differences
    float maxUDif = 0;
    float avgUDif = 0;
    float totalUDif = 0;
    float uDif = 0;
    for(size_t uDifIndex = 0; uDifIndex < WGD->u.size(); uDifIndex++){
      uDif = std::abs(WGD->u[uDifIndex] - completedSolvers[solversIndex]->u[uDifIndex]);
      if(uDif>maxUDif) maxUDif = uDif;
      totalUDif += uDif;
    }
    avgUDif = totalUDif/WGD->u.size();

    // Calculating v differences
    float maxVDif = 0;
    float avgVDif = 0;
    float totalVDif = 0;
    float vDif = 0;
    for(size_t vDifIndex = 0; vDifIndex < WGD->v.size(); vDifIndex++){
      vDif = std::abs(WGD->v[vDifIndex] - completedSolvers[solversIndex]->v[vDifIndex]);
      if(vDif>maxVDif) maxVDif = vDif;
      totalVDif += vDif;
    }
    avgVDif = totalVDif/WGD->v.size();

    // Calculating w differences
    float maxWDif = 0;
    float avgWDif = 0;
    float totalWDif = 0;
    float wDif = 0;
    for(size_t wDifIndex = 0; wDifIndex < WGD->w.size(); wDifIndex++){
      wDif = std::abs(WGD->w[wDifIndex] - completedSolvers[solversIndex]->w[wDifIndex]);
      if(wDif>maxWDif) maxWDif = wDif;
      totalWDif += wDif;
    }
    avgWDif = totalWDif/WGD->w.size();

    // Wind velocity magnitude (WindVelMag) difference calculations
    float cpuWVM = 0;
    float gpuWVM = 0;
    float totalWvmDif = 0;
    float avgWvmDif = 0;
    float maxWvmDif = 0;
    float cpuSum = 0;
    float gpuSum = 0;
    // Calculate vector magnitudes, find difference and then add to sum
    for(size_t i = 0; i < WGD->w.size(); ++i){
      cpuWVM = sqrt(((WGD->u[i])*(WGD->u[i]))+
		    ((WGD->v[i])*(WGD->v[i]))+
		    ((WGD->w[i])*(WGD->w[i])));
      gpuWVM  = sqrt(((completedSolvers[solversIndex]->u[i])*(completedSolvers[solversIndex]->u[i]))+
			((completedSolvers[solversIndex]->v[i])*(completedSolvers[solversIndex]->v[i]))+
                        ((completedSolvers[solversIndex]->w[i])*(completedSolvers[solversIndex]->w[i])));
      totalWvmDif += std::abs(cpuWVM-gpuWVM);
      if(std::abs(cpuWVM-gpuWVM) > maxWvmDif) maxWvmDif = std::abs(cpuWVM-gpuWVM);
      // These sums required for R-squared calculations below
      cpuSum += cpuWVM;
      gpuSum += gpuWVM;
    }
    avgWvmDif = totalWvmDif/WGD->w.size();
    // /////////////////////////
    // CALCULATING R-squared
    // /////////////////////////

    // R-squared is calculated by first finding r (shocking), and squaring it.
    // To find r, you find the standard deviation of the two data sets,
    // then find the covariance between them. Divide the covariance by the
    // product of the two standard deviations, and you'll find R.
    float cpuAverage = cpuSum/WGD->w.size();
    float gpuAverage = gpuSum/completedSolvers[solversIndex]->w.size();
    float cpuDevSum = 0;
    float gpuDevSum = 0;
    float solversCovariance = 0;
    float solversCovarianceSum = 0;
    for(size_t i = 0; i < WGD->w.size(); i++){
      // Calculate vector magnitudes
      cpuWVM = sqrt(((WGD->u[i])*(WGD->u[i]))+
                    ((WGD->v[i])*(WGD->v[i]))+
                    ((WGD->w[i])*(WGD->w[i])));
      gpuWVM  = sqrt(((completedSolvers[solversIndex]->u[i])*(completedSolvers[solversIndex]->u[i]))+
                        ((completedSolvers[solversIndex]->v[i])*(completedSolvers[solversIndex]->v[i]))+
                        ((completedSolvers[solversIndex]->w[i])*(completedSolvers[solversIndex]->w[i])));
      // Calculate deviation and covariance sums
      cpuDevSum += (cpuWVM-cpuAverage)*(cpuWVM-cpuAverage);
      gpuDevSum += (gpuWVM-gpuAverage)*(gpuWVM-gpuAverage);
      solversCovarianceSum += (cpuWVM-cpuAverage) * (gpuWVM-gpuAverage);
    }
    // Calculate standard deviations and covariance
    float cpuStDev = std::sqrt(cpuDevSum/(WGD->w.size()-1));
    float gpuStDev = std::sqrt(gpuDevSum/(WGD->w.size()-1));
    solversCovariance = solversCovarianceSum/(WGD->w.size()-1);
    // Calculate r and r^2
    float r = solversCovariance/(cpuStDev*gpuStDev);
    float rSquared = r * r;

    // Table comparison metrics row
    t.add(solverNames[solversIndex]);
    t.add(std::to_string(maxUDif));
    t.add(std::to_string(maxVDif));
    t.add(std::to_string(maxWDif));
    t.add(std::to_string(avgUDif));
    t.add(std::to_string(avgVDif));
    t.add(std::to_string(avgWDif));
    t.add(std::to_string(totalWvmDif));
    t.add(std::to_string(rSquared));
    t.add(std::to_string(avgWvmDif));
    t.add(std::to_string(maxWvmDif));
    t.endOfRow();

    // Stores results for bar chart construction that happens after this loop completes
    avgWVMResults.push_back(avgWvmDif);
    wvmResults.push_back(totalWvmDif);
    rSquaredResults.push_back(rSquared);
   
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
  std::cout << t << std::endl;
  // Print comparison bar charts if user is running all solvers
  if(arguments.solveType == 1) {
    printChart("WindVelMag DIFF", false, wvmResults);
    printChart("Average WVM DIFF", false, avgWVMResults);
    //printChart("R^2 VALUE", true, rSquaredResults);
  }
  std::cout << "✔ Comparative analysis complete\n";

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
WINDSGeneralData* runSerial(WINDSGeneralData* WGD, WINDSInputData* WID, Solver *solverCPU, bool solveWind){
  std::cout << std::endl;
  std::cout << "Run Serial Solver (CPU)..." << std::endl;
  solverCPU = new CPUSolver(WID, WGD);
  solverCPU->solve(WID, WGD, !solveWind);
  std::cout << "✔ CPU solver done\n";
  std::cout << std::endl;
  return WGD;
}

//Dynamic parallel GPU solver
WINDSGeneralData* runDynamic(WINDSGeneralData* WGD_DYNAMIC, WINDSInputData* WID, Solver *solverDynamic, bool solveWind){
  std::cout << "Run Dynamic Parallel Solver (GPU)..." << std::endl;
  solverDynamic = new DynamicParallelism(WID, WGD_DYNAMIC);
  solverDynamic->solve(WID, WGD_DYNAMIC, !solveWind);
  std::cout << "✔ Dynamic solver done\n";
  std::cout << std::endl;
  return WGD_DYNAMIC;
}

//Global memory GPU solver
WINDSGeneralData* runGlobal(WINDSGeneralData* WGD_GLOBAL, WINDSInputData* WID, Solver *solverGlobal, bool solveWind){
  std::cout << "Run Global Memory Solver (GPU)..." << std::endl;
  solverGlobal = new GlobalMemory(WID, WGD_GLOBAL);
  solverGlobal->solve(WID, WGD_GLOBAL, !solveWind);
  std::cout << "✔ Global solver done\n";
  std::cout << std::endl;
  return WGD_GLOBAL;
}

//Shared memory GPU solver
WINDSGeneralData* runShared(WINDSGeneralData* WGD_SHARED, WINDSInputData* WID, Solver *solverShared, bool solveWind){
  std::cout << "Run Shared Memory Solver (GPU)..." << std::endl;
  solverShared = new SharedMemory(WID, WGD_SHARED);
  solverShared->solve(WID, WGD_SHARED, !solveWind);
  std::cout << "✔ Shared solver done\n";
  std::cout << std::endl;
  return WGD_SHARED;
}

void printChart(string title, bool isHigherBetter, vector<double> results)
{
  // Windows needs a little help...
  #ifdef _WIN32
  SetConsoleOutputCP( CP_UTF8 );
  #endif

  // Maximum screen height
  int max_lines = 12;

  // Calculate the vertical dimensions of the graph,
  // fitting it to the available space as necessary.
  double domain = *std::max_element(results.begin(), results.end());
  double divisor = (domain < (max_lines - 3)) ? 1 : domain / (max_lines - 3);
  int nlines = (domain < (max_lines - 3)) ? (int)domain : (max_lines - 3);

  // Output Glyphs (this is the UTF-8 required part)
  const char* none_bar   = "────";
  const char* half_bar   = "─▄▄▄";
  const char* full_bar   = "─███";
  const char* x_axis     = "─▀▀▀";
  const char* x_axis_0   = "────";
  const char* x_axis_cap = "─ ";
  const char* y_axis     = "├";
  const char* line_cap   = "─ ";
  const char* origin     = "└";

  // Find max number length for future vertical alignment calculations
  int maxNum = *max_element(results.begin(), results.end());
  int maxNumLength = std::to_string(maxNum).length();
  int spaces = 0;

  // Draw the graph
  spaces = maxNumLength;
  std::cout << std::string(spaces, ' ') << title << std::endl;
  if (isHigherBetter)
    std::cout << std::string(spaces, ' ') << "HIGHER IS BETTER";
  else
    std::cout << std::string(spaces, ' ') << "LOWER IS BETTER";

  // Draw everything above the X-axis
  std::cout << std::fixed << "\n";
  for (int n = 0; n < nlines; n++)
  {
    double y2 = (nlines - n) * divisor;
    double y1 = y2 - divisor / 4;
    // Calculate number of spaces to add before Y-axis labels
    int y2Length = std::to_string((int)y2).length();
    spaces = maxNumLength - y2Length;
    std::cout << std::string(spaces, ' ') << std::setprecision(0) << y2 << y_axis;
    for (auto x : results)
    {
      if (x > y2) std::cout << full_bar;
      else if (x > y1) std::cout << half_bar;
      else std::cout << none_bar;
    }
    std::cout << line_cap << "\n";
  }

  // Draw the X-axis
  spaces = maxNumLength - 1;
  std::cout << std::string(spaces, ' ') << "0" << origin;
  for (auto x : results)
  {
    if (x > divisor / 4) std::cout << x_axis;
    else std::cout << x_axis_0;
  }
  std::cout << x_axis_cap << "\n";

  // Draw labels under the X-axis
  spaces = maxNumLength + 2;
  std::cout << std::string(spaces, ' ');
  std::vector<string> xAxisLabels = {"DYN","GBL","SHR"};
  for (int n = 0; n < xAxisLabels.size(); ++n)
    std::cout << std::setw(4) << xAxisLabels[n];
  std::cout << "\n" << std::endl;
}
