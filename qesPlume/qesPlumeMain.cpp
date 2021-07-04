
#include <iostream>
#include <netcdf>
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>


#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "util/calcTime.h"


#include "src/plume/Args.hpp"
#include "src/plume/PlumeInputData.hpp"
#include "util/NetCDFInput.h"

#include "src/winds/WINDSGeneralData.h"
#include "src/winds/TURBGeneralData.h"

#include "src/plume/Plume.hpp"
#include "src/plume/Eulerian.h"

#include "util/QESNetCDFOutput.h"
#include "src/plume/PlumeOutputEulerian.h"
#include "src/plume/PlumeOutput.h"
#include "src/plume/PlumeOutputParticleData.h"


// LA do these need to be here???
using namespace netCDF;
using namespace netCDF::exceptions;

PlumeInputData *parseXMLTree(const std::string fileName);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  // set up timer information for the simulation runtime
  calcTime timers;
  timers.startNewTimer("QES-Plume total runtime");// start recording execution time


  // print a nice little welcome message
  std::cout << std::endl;
  std::cout << "##############################################################" << std::endl;
  std::cout << "#                                                            #" << std::endl;
  std::cout << "#                   Welcome to QES-PLUME                     #" << std::endl;
  std::cout << "#                                                            #" << std::endl;
  std::cout << "##############################################################" << std::endl;

  // parse command line arguments
  Args arguments;
  arguments.processArguments(argc, argv);

  // parse xml settings
  PlumeInputData *PID = new PlumeInputData(arguments.inputQESFile);

  //PlumeInputData* PID = parseXMLTree(arguments.inputQESFile);
  //if ( !PID ) {
  //    std::cerr << "QES-Plume input file: " << arguments.inputQESFile << " not able to be read successfully." << std::endl;
  //    exit(EXIT_FAILURE);
  //}

  // Create instance of QES-winds General data class
  WINDSGeneralData *WGD = new WINDSGeneralData(arguments.inputWINDSFile);
  // Create instance of QES-Turb General data class
  TURBGeneralData *TGD = new TURBGeneralData(arguments.inputTURBFile, WGD);
  // Create instance of Eulerian class
  Eulerian *eul = new Eulerian(PID, WGD, TGD, arguments.debug);

  //load data at t=0;
  TGD->loadNetCDFData(0);
  WGD->loadNetCDFData(0);
  eul->setData(WGD, TGD);

  // Create instance of Plume model class
  Plume *plume = new Plume(PID, WGD, TGD, eul, &arguments);

  // create output instance
  std::vector<QESNetCDFOutput *> outputVec;
  // always supposed to output lagrToEulOutput data
  outputVec.push_back(new PlumeOutput(PID, WGD, plume, arguments.outputFile));
  if (arguments.doParticleDataOutput == true) {
    outputVec.push_back(new PlumeOutputParticleData(PID, plume, arguments.outputParticleDataFile));
  }

  // create output instance (separate for eulerian class)
  QESNetCDFOutput *eulOutput = nullptr;
  if (arguments.doEulDataOutput == true) {
    eulOutput = new PlumeOutputEulerian(PID, WGD, TGD, eul, arguments.outputEulerianFile);
    // output Eulerian data. Use time zero
    eulOutput->save(0.0);
  }

  // Run plume advection model
  plume->run(PID->simParams->simDur, WGD, TGD, eul, outputVec);

  // compute run time information and print the elapsed execution time
  std::cout << "[QES-Plume] \t Finished. \n"
            << std::endl;
  std::cout << "End run particle summary \n";
  plume->showCurrentStatus();
  timers.printStoredTime("QES-Plume total runtime");
  std::cout << "##############################################################" << std::endl;

  exit(EXIT_SUCCESS);
}

PlumeInputData *parseXMLTree(const std::string fileName)
{
  pt::ptree tree;

  try {
    pt::read_xml(fileName, tree);
  } catch (boost::property_tree::xml_parser::xml_parser_error &e) {
    std::cerr << "Error reading tree in" << fileName << "\n";
    return (PlumeInputData *)0;
  }

  PlumeInputData *xmlRoot = new PlumeInputData();
  xmlRoot->parseTree(tree);
  return xmlRoot;
}
