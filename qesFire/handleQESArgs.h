/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/** 
 * @file handleQESArgs.h
 * @class QESArgs
 * @brief This class handles different commandline options and arguments
 * and places the values into variables.
 * @note Child of ArgumentParsing
 * @sa ArgumentParsing
 */
#pragma once

#include <iostream>
#include "util/ArgumentParsing.h"

#include "util/QESout.h"
#include "winds/WINDSGeneralData.h"
#include "plume/PLUMEGeneralData.h"

enum solverTypes : int { CPU_Type = 1,
                         DYNAMIC_P = 2,
                         Global_M = 3,
                         Shared_M = 4 };

class QESArgs : public ArgumentParsing
{
public:
  QESArgs();

  ~QESArgs() {}

  /*
     * Takes in the commandline arguments and places
     * them into variables.
     *
     * @param argc -number of commandline options/arguments
     * @param argv -array of strings for arguments
     */
  void processArguments(int argc, char *argv[]);


  bool verbose;

  // input files (from the command line)
  std::string qesWindsParamFile = "";
  std::string qesPlumeParamFile = "";

  // Base name for all NetCDF output files
  std::string netCDFFileBasename = "";

  // flag to turn on/off different modules
  bool solveWind, compTurb, compPlume, fireWindsFlag;
  int solveType;

  // QES_WINDS output files:
  bool visuOutput, wkspOutput, terrainOut;
  // netCDFFile for standard cell-center vizalization file
  std::string netCDFFileVisu = "";
  // netCDFFile for working field used by Plume
  std::string netCDFFileWksp = "";
  // filename for terrain output
  std::string filenameTerrain = "";

  // QES-TURB output files:
  bool turbOutput;
  // netCDFFile for turbulence field used by Plume
  std::string netCDFFileTurb = "";

  // QES-Plume output files
  PlumeParameters plumeParameters;
  // going to assume concentration is always output. So these next options are like choices for additional debug output
  bool plumeOutput, particleOutput;

  // output file variables created from the outputFolder and caseBaseName
  //std::string outputEulerianFile;
  std::string outputPlumeFile;
  std::string outputParticleDataFile;

  // QES-Fire output files
  bool fireOutput;
  std::string netCDFFileFireOut = "";

private:
};
