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
  // reg("visuout", "Turns on the netcdf file to write visulatization results", ArgumentParsing::NONE, 'z');
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

  std::cout << "Summary of QES-WINDS options: " << std::endl;
  std::cout << "----------------------------" << std::endl;

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

  if (compTurb) {
    std::cout << "Turbulence model:\t ON" << std::endl;
  } else {
    std::cout << "Turbulence model:\t OFF" << std::endl;
  }

  if (verbose) {
    std::cout << "Verbose:\t\t ON" << std::endl;
  } else {
    std::cout << "Verbose:\t\t OFF" << std::endl;
  }

  std::cout << "----------------------------" << std::endl;

  std::cout << "qesWindsParamFile set to " << qesWindsParamFile << std::endl;

  std::cout << "----------------------------" << std::endl;

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

    if (!netCDFFileVisu.empty()) {
      std::cout << "[WINDS]\t Visualization NetCDF output file set:\t " << netCDFFileVisu << std::endl;
    }
    if (!netCDFFileWksp.empty()) {
      std::cout << "[WINDS]\t Workspace NetCDF output file set:\t " << netCDFFileWksp << std::endl;
    }
    if (!filenameTerrain.empty()) {
      std::cout << "[WINDS]\t Terrain triangle mesh output set:\t " << filenameTerrain << std::endl;
    }
    if (!netCDFFileTurb.empty()) {
      std::cout << "[TURB]\t Turbulence NetCDF output file set:\t " << netCDFFileTurb << std::endl;
    }
    netCDFFileFire = netCDFFileBasename;
    netCDFFileFire.append("_fireOut.nc");
  } else {

    QESout::warning("No output basename set -> output turned off ");
    visuOutput = false;
    wkspOutput = false;
    turbOutput = false;
    terrainOut = false;
  }

  std::cout << "###################################################################" << std::endl;
}
