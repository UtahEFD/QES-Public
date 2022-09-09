/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file handlWINDSArgs.h */

#pragma once

#include <iostream>
#include "util/ArgumentParsing.h"
#include "util/QEStool.h"

enum solverTypes : int { CPU_Type = 1,
                         DYNAMIC_P = 2,
                         Global_M = 3,
                         Shared_M = 4 };

/**
 * @class WINDSArgs
 * @brief Handles different commandline options and arguments
 * and places the values into variables.
 *
 * @note Child of ArgumentParsing
 * @sa ArgumentParsing
 */
class WINDSArgs : public ArgumentParsing
{
public:
  WINDSArgs();

  ~WINDSArgs() {}

  /**
   * Takes in the commandline arguments and places
   * them into variables.
   *
   * @param argc Number of commandline options/arguments
   * @param argv Array of strings for arguments
   */
  void processArguments(int argc, char *argv[]);


  bool verbose;


  std::string qesWindsParamFile = ""; /**< Input files (from cmd line) */

  std::string netCDFFileBasename = ""; /**< Base name for all NetCDF output files */

  ///@{
  /** Flag to turn on/off different modules */
  bool solveWind, compTurb;
  int solveType, compareType;
  bool visuOutput, wkspOutput, turbOutput, terrainOut;
  ///@}

  std::string netCDFFileVisu = ""; /**< netCDFFile for standard cell-center visualization file */
  std::string netCDFFileWksp = ""; /**< netCDFFile for working field used by Plume */
  std::string netCDFFileTurb = ""; /**< netCDFFile for turbulence field used by Plume */
  std::string filenameTerrain = ""; /**< Filename for terrain output */

  bool fireMode; /**< Boolean to treat WRF input in fire mode */

private:
};
