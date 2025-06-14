/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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


// #include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "util/calcTime.h"


#include "handlePlumeArgs.h"
#include "plume/PLUMEInputData.h"
#include "util/NetCDFInput.h"
#include "util/QESout.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "plume/PLUMEGeneralData.h"

#include "util/QESNetCDFOutput.h"


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
  std::cout << "WGD->UTMx:   " << WGD->UTMx << std::endl;
  // Create instance of QES-Turb General data class
  TURBGeneralData *TGD = new TURBGeneralData(arguments.inputTURBFile, WGD);
  // Create instance of QES-Plume General data class
  PLUMEGeneralData *PGD = new PLUMEGeneralData(arguments.plumeParameters, PID, WGD, TGD);

  for (int index = 0; index < WGD->totalTimeIncrements; index++) {
    // Load data at current time index
    WGD->loadNetCDFData(index);
    TGD->loadNetCDFData(index);

    // Determine the end time for advection
    QEStime endTime = WGD->nextTimeInstance(index, PID->plumeParams->simDur);

    // Run plume advection model
    PGD->run(endTime, WGD, TGD);

    std::cout << "[QES-Plume] \t Finished." << std::endl;
  }

  PGD->showCurrentStatus();

  std::cout << "##############################################################" << std::endl;

  delete WGD;
  delete TGD;

  delete PID;
  delete PGD;

  exit(EXIT_SUCCESS);
}
