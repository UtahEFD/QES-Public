/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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

/**
 * @file handleWINDSArgs.cpp
 * @brief Handles different commandline options and arguments
 * and places the values into variables.
 *
 * @note Child of ArgumentParsing
 * @sa ArgumentParsing
 */

#include "handleWINDSArgs.h"

WINDSArgs::WINDSArgs()
  : verbose(false),
    qesWindsParamFile(""), netCDFFileBasename(""),
    compTurb(false), solveType(1),
    visuOutput(true), wkspOutput(false), turbOutput(false), terrainOut(false),
    fireMode(false)
{
  reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');

  reg("windsolveroff", "Turns off the wind solver and wind output", ArgumentParsing::NONE, 'x');
  reg("solvetype", "selects the method for solving the windfield", ArgumentParsing::INT, 's');

  reg("qesWindsParamFile", "Specifies input xml settings file", ArgumentParsing::STRING, 'q');

  reg("outbasename", "Specifies the basename for netcdf files", ArgumentParsing::STRING, 'o');
  //reg("visuout", "Turns on the netcdf file to write visulatization results", ArgumentParsing::NONE, 'z');
  reg("wkout", "Turns on the netcdf file to write wroking file", ArgumentParsing::NONE, 'w');
  // [FM] the output of turbulence field linked to the flag compTurb
  // reg("turbout", "Turns on the netcdf file to write turbulence file", ArgumentParsing::NONE, 'r');
  reg("terrainout", "Turn on the output of the triangle mesh for the terrain", ArgumentParsing::NONE, 'h');

  reg("turbcomp", "Turns on the computation of turbulent fields", ArgumentParsing::NONE, 't');
  reg("firemode", "Enable writing of wind data back to WRF Input file.", ArgumentParsing::NONE, 'f');
}

void WINDSArgs::processArguments(int argc, char *argv[])
{
  processCommandLineArgs(argc, argv);

  // Process the command line arguments after registering which
  // arguments you wish to parse.
  if (isSet("help")) {
    printUsage();
    exit(EXIT_SUCCESS);
  }

  verbose = isSet("verbose");
  if (verbose) std::cout << "Verbose Output: ON" << std::endl;

  solveWind = isSet("windsolveroff");
  if (solveWind) std::cout << "the wind fields are not being calculated" << std::endl;

  isSet("solvetype", solveType);
  if (solveType == CPU_Type)
    std::cout << "Solving with: Serial solver (CPU)" << std::endl;
  else if (solveType == DYNAMIC_P)
    std::cout << "Solving with: Dynamic Parallel solver (GPU)" << std::endl;
  else if (solveType == Global_M)
    std::cout << "Solving with: Global memory solver (GPU)" << std::endl;
  else if (solveType == Shared_M)
    std::cout << "Solving with: Shared memory solver (GPU)" << std::endl;

  compTurb = isSet("turbcomp");
  if (compTurb) std::cout << "Turbulence model: ON" << std::endl;

  fireMode = isSet("firemode");
  if (fireMode) std::cout << "Wind data will be written back to WRF input file." << std::endl;

  isSet("qesWindsParamFile", qesWindsParamFile);
  if (qesWindsParamFile != "")
    std::cout << "qesWindsParamFile set to " << qesWindsParamFile << std::endl;
  else
    QEStool::error("qesWindsParamFile not specified");

  isSet("outbasename", netCDFFileBasename);
  if (netCDFFileBasename != "") {
    //visuOutput = isSet("visuout");
    if (visuOutput) {
      netCDFFileVisu = netCDFFileBasename;
      netCDFFileVisu.append("_windsOut.nc");
      std::cout << "Visualization NetCDF output file set to " << netCDFFileVisu << std::endl;
    }

    wkspOutput = isSet("wkout");
    if (wkspOutput) {
      netCDFFileWksp = netCDFFileBasename;
      netCDFFileWksp.append("_windsWk.nc");
      std::cout << "Workspace NetCDF output file set to " << netCDFFileWksp << std::endl;
    }

    // [FM] the output of turbulence field linked to the flag compTurb
    // -> subject to change
    turbOutput = compTurb;// isSet("turbout");
    if (turbOutput) {
      netCDFFileTurb = netCDFFileBasename;
      netCDFFileTurb.append("_turbOut.nc");
      std::cout << "Turbulence NetCDF output file set to " << netCDFFileTurb << std::endl;
    }

    terrainOut = isSet("terrainout");
    if (terrainOut) {
      filenameTerrain = netCDFFileBasename;
      filenameTerrain.append("_terrainOut.obj");
      std::cout << "Terrain triangle mesh WILL be output to " << filenameTerrain << std::endl;
    }

  } else {
    QEStool::warning("No output basename set -> output turned off ");
    visuOutput = false;
    wkspOutput = false;
    turbOutput = false;
    terrainOut = false;
  }
}
