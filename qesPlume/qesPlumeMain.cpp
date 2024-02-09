/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
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

#include "util/calcTime.h"


#include "handlePlumeArgs.hpp"
#include "plume/PlumeInputData.hpp"
#include "util/NetCDFInput.h"
#include "util/QESout.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "plume/PLUMEGeneralData.h"

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
  QESout::splashScreen();

  // parse command line arguments
  PlumeArgs arguments;
  arguments.processArguments(argc, argv);

  // parse xml settings
  PlumeInputData *PID = new PlumeInputData(arguments.qesPlumeParamFile);
  if (!PID)
    QESout::error("QES-Plume input file: " + arguments.qesPlumeParamFile + " not able to be read successfully.");

  // Create instance of QES-winds General data class
  WINDSGeneralData *WGD = new WINDSGeneralData(arguments.inputWINDSFile);
  // Create instance of QES-Turb General data class
  TURBGeneralData *TGD = new TURBGeneralData(arguments.inputTURBFile, WGD);

  // Create instance of Plume model class
  PLUMEGeneralData *PGD = new PLUMEGeneralData(PID, WGD, TGD);

  // create output instance
  std::vector<QESNetCDFOutput *> outputVec;
  // always supposed to output lagrToEulOutput data
  outputVec.push_back(new PlumeOutput(PID, PGD, arguments.outputFile));
  if (arguments.doParticleDataOutput) {
    // outputVec.push_back(new PlumeOutputParticleData(PID, PGD, arguments.outputParticleDataFile));
  }

  for (int index = 0; index < WGD->totalTimeIncrements; index++) {
    // load data at
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
    PGD->run(endtime, WGD, TGD, outputVec);

    // compute run time information and print the elapsed execution time
    std::cout << "[QES-Plume] \t Finished." << std::endl;
  }

  PGD->showCurrentStatus();
  std::cout << "##############################################################" << std::endl;

  exit(EXIT_SUCCESS);
}
