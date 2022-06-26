
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


#include "plume/handlePlumeArgs.hpp"
#include "plume/PlumeInputData.hpp"
#include "util/NetCDFInput.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "plume/Plume.hpp"

#include "util/QESNetCDFOutput.h"
#include "plume/PlumeOutput.h"
#include "plume/PlumeOutputParticleData.h"


// LA do these need to be here???
using namespace netCDF;
using namespace netCDF::exceptions;

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
  PlumeArgs arguments;
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

  // Create instance of Plume model class
  Plume *plume = new Plume(PID, WGD, TGD);

  // create output instance
  std::vector<QESNetCDFOutput *> outputVec;
  // always supposed to output lagrToEulOutput data
  outputVec.push_back(new PlumeOutput(PID, WGD, plume, arguments.outputFile));
  if (arguments.doParticleDataOutput == true) {
    outputVec.push_back(new PlumeOutputParticleData(PID, plume, arguments.outputParticleDataFile));
  }

  for (int index = 0; index < WGD->totalTimeIncrements; index++) {
    //load data at
    TGD->loadNetCDFData(index);
    WGD->loadNetCDFData(index);

    // Run plume advection model
    QEStime endtime;
    if (WGD->totalTimeIncrements == 1) {
      endtime = WGD->timestamp[index] + PID->plumeParams->simDur;
    } else if (index == WGD->totalTimeIncrements - 1) {
      endtime = WGD->timestamp[index] + (WGD->timestamp[index] - WGD->timestamp[index - 1]);
    } else {
      endtime = WGD->timestamp[index + 1];
    }
    plume->run(endtime, WGD, TGD, outputVec);

    // compute run time information and print the elapsed execution time
    std::cout << "[QES-Plume] \t Finished." << std::endl;
  }

  std::cout << "End run particle summary \n";
  plume->showCurrentStatus();
  timers.printStoredTime("QES-Plume total runtime");
  std::cout << "##############################################################" << std::endl;

  exit(EXIT_SUCCESS);
}
