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
 * This file is part of QES
 *
 * GPL-3.0 License
 *
 * QES is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

#pragma once

/*
 * This class handles different commandline options and arguments
 * and places the values into variables. This inherits from Argument Parsing
 */

#include <iostream>
#include "util/ArgumentParsing.h"
#include "util/QESout.h"

#include "winds/SolverFactory.h"
#include "winds/WINDSGeneralData.h"

#include "plume/PLUMEGeneralData.h"

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
  std::string qesWindsParamFile;
  std::string qesPlumeParamFile;

  // Base name for all NetCDF output files
  std::string outputFileBasename;

  // flag to turn on/off different modules
  bool solveWind, compTurb, compPlume;
  int solveType;

  // QES_WINDS output files:
  bool visuOutput, wkspOutput, terrainOut;
  // netCDFFile for standard cell-center vizalization file
  std::string netCDFFileVisu;
  // netCDFFile for working field used by Plume
  std::string netCDFFileWksp;
  // filename for terrain output
  std::string filenameTerrain;

  // QES-TURB output files:
  bool turbOutput;
  // netCDFFile for turbulence field used by Plume
  std::string netCDFFileTurb;

  // QES-Plume output files
  PlumeParameters plumeParameters;
  // going to assume concentration is always output. So these next options are like choices for additional debug output
  bool plumeOutput, particleOutput;

  // output file variables created from the outputFolder and caseBaseName
  // std::string outputEulerianFile;
  std::string outputPlumeFile;
  std::string outputParticleDataFile;

private:
};
