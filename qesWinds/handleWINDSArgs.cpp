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
    compTurb(false), solveWind(true), solveType(1),
    visuOutput(true), wkspOutput(false), turbOutput(false), terrainOut(false),
    fireMode(false)
{
  reg("help", "help/usage information", ArgumentParsing::NONE, 'h');
  reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');

  reg("windsolveroff", "Turns off the wind solver and wind output", ArgumentParsing::NONE, 'x');
  reg("solvetype", "selects the method for solving the windfield", ArgumentParsing::INT, 's');

  reg("qesWindsParamFile", "Specifies input xml settings file", ArgumentParsing::STRING, 'q');

  reg("outbasename", "Specifies the basename for netcdf files", ArgumentParsing::STRING, 'o');
  // reg("visuout", "Turns on the netcdf file to write visulatization results", ArgumentParsing::NONE, 'z');
  reg("wkout", "Turns on the netcdf file to write wroking file", ArgumentParsing::NONE, 'w');
  // [FM] the output of turbulence field linked to the flag compTurb
  // reg("turbout", "Turns on the netcdf file to write turbulence file", ArgumentParsing::NONE, 'r');
  reg("terrainout", "Turn on the output of the triangle mesh for the terrain", ArgumentParsing::NONE, 'm');

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

  isSet("qesWindsParamFile", qesWindsParamFile);
  if (qesWindsParamFile.empty()) {
    QESout::error("qesWindsParamFile not specified");
  }

  solveWind = !isSet("windsolveroff");
  isSet("solvetype", solveType);
#ifndef HAS_CUDA
  // if CUDA is not supported, force the solveType to be CPU no matter
  solveType = CPU_Type;
#endif

  compTurb = isSet("turbcomp");
  fireMode = isSet("firemode");
  if (fireMode) std::cout << "Wind data will be written back to WRF input file." << std::endl;

  verbose = isSet("verbose");
  if (verbose) {
    QESout::setVerbose();
  }

  isSet("outbasename", netCDFFileBasename);
  if (!netCDFFileBasename.empty()) {
    // visuOutput = isSet("visuout");
    if (visuOutput) {
      netCDFFileVisu = netCDFFileBasename;
      netCDFFileVisu.append("_windsOut.nc");
    }

    wkspOutput = isSet("wkout");
    if (wkspOutput) {
      netCDFFileWksp = netCDFFileBasename;
      netCDFFileWksp.append("_windsWk.nc");
    }

    // [FM] the output of turbulence field linked to the flag compTurb
    // -> subject to change
    turbOutput = compTurb;// isSet("turbout");
    if (turbOutput) {
      netCDFFileTurb = netCDFFileBasename;
      netCDFFileTurb.append("_turbOut.nc");
    }

    terrainOut = isSet("terrainout");
    if (terrainOut) {
      filenameTerrain = netCDFFileBasename;
      filenameTerrain.append("_terrainOut.obj");
    }

    netCDFFileFire = netCDFFileBasename;
    netCDFFileFire.append("_fireOut.nc");
  } else {
    visuOutput = false;
    wkspOutput = false;
    turbOutput = false;
    terrainOut = false;
  }


  std::cout << "Summary of QES-WINDS options: " << std::endl;
  std::cout << "----------------------------" << std::endl;
  // parameter files:
  std::cout << "qesWindsParamFile: " << qesWindsParamFile << std::endl;

  std::cout << "----------------------------" << std::endl;

  // code options:
  if (solveWind) {
    if (solveType == CPU_Type)
#ifdef _OPENMP
      std::cout << "Wind Solver:\t\t ON\t [Red/Black Solver (CPU)]" << std::endl;
#else
      std::cout << "Wind Solver:\t\t ON\t [Serial solver (CPU)]" << std::endl;
#endif
    else if (solveType == DYNAMIC_P)
      std::cout << "Wind Solver:\t\t ON\t [Dynamic Parallel solver (GPU)]" << std::endl;
    else if (solveType == Global_M)
      std::cout << "Wind Solver:\t\t ON\t [Global memory solver (GPU)]" << std::endl;
    else if (solveType == Shared_M)
      std::cout << "Wind Solver:\t\t ON\t [Shared memory solver (GPU)]" << std::endl;
    else
      std::cout << "[WARNING]\t the wind fields are not being calculated" << std::endl;
  }
  std::cout << "Turbulence model:\t " << (compTurb ? "ON" : "OFF") << std::endl;
  std::cout << "Verbose:\t\t " << (verbose ? "ON" : "OFF") << std::endl;

  // output files:
  if (!netCDFFileBasename.empty()) {
    std::cout << "----------------------------" << std::endl;
    std::cout << "Output file basename:        " << netCDFFileBasename << std::endl;
    std::cout << "Winds visualization output:  " << (visuOutput ? "ON" : "OFF") << std::endl;
    std::cout << "Winds workspace output:      " << (wkspOutput ? "ON" : "OFF") << std::endl;
    std::cout << "Winds terrain mesh output:   " << (terrainOut ? "ON" : "OFF") << std::endl;
    std::cout << "Turbulence output:           " << (turbOutput ? "ON" : "OFF") << std::endl;
  } else {
    QESout::warning("No output basename set -> output turned off ");
  }

  std::cout << "###################################################################" << std::endl;
}
