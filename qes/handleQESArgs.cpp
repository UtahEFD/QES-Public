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

#include "handleQESArgs.h"

QESArgs::QESArgs()
  : verbose(false),
    solveWind(false), compTurb(false), compPlume(false),
    solveType(1),
    visuOutput(true), wkspOutput(false), terrainOut(false),
    turbOutput(false),
    plumeOutput(false), particleOutput(false)

{
  // reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  reg("help", "help/usage information", ArgumentParsing::NONE, 'h');

  reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');

  reg("windsolveroff", "Turns off the wind solver and wind output", ArgumentParsing::NONE, 'x');
  reg("solvetype", "selects the method for solving the windfield", ArgumentParsing::INT, 's');

  reg("qesWindsParamFile", "Specifies the QES Proj file", ArgumentParsing::STRING, 'q');
  reg("qesPlumeParamFile", "Specifies the QES Proj file", ArgumentParsing::STRING, 'p');

  reg("turbcomp", "Turns on the computation of turbulent fields", ArgumentParsing::NONE, 't');

  reg("outbasename", "Specifies the basename for netcdf files", ArgumentParsing::STRING, 'o');
  reg("visuout", "Turns on the netcdf file to write visulatization results", ArgumentParsing::NONE, 'z');
  reg("wkout", "Turns on the netcdf file to write wroking file", ArgumentParsing::NONE, 'w');
  // [FM] the output of turbulence field linked to the flag compTurb
  // reg("turbout", "Turns on the netcdf file to write turbulence file", ArgumentParsing::NONE, 'r');
  reg("terrainout", "Turn on the output of the triangle mesh for the terrain", ArgumentParsing::NONE, 'm');

  // going to assume concentration is always output. So these next options are like choices for additional debug output
  // reg("doEulDataOutput",     "should debug Eulerian data be output",           ArgumentParsing::NONE,   'e');
  reg("particleOutput", "should debug Lagrangian data be output", ArgumentParsing::NONE, 'l');
}


void QESArgs::processArguments(int argc, char *argv[])
{
  processCommandLineArgs(argc, argv);

  // Process the command line arguments after registering which
  // arguments you wish to parse.
  if (isSet("help")) {
    printUsage();
    exit(EXIT_SUCCESS);
  }

  verbose = isSet("verbose");
  if (verbose) {
    QESout::setVerbose();
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
  compPlume = isSet("qesPlumeParamFile", qesPlumeParamFile);
  if (compPlume) {
    if (compTurb) {
      compTurb = true;
      turbOutput = true;
    } else {
      compTurb = true;
      turbOutput = false;
    }
    plumeOutput = true;
  } else {
    turbOutput = compTurb;
  }

  particleOutput = isSet("particleOutput");

  isSet("outbasename", outputFileBasename);
  if (!outputFileBasename.empty()) {
    // visuOutput = isSet("visuout");
    if (visuOutput) {
      netCDFFileVisu = outputFileBasename;
      netCDFFileVisu.append("_windsOut.nc");
    }

    wkspOutput = isSet("wkout");
    if (wkspOutput) {
      netCDFFileWksp = outputFileBasename;
      netCDFFileWksp.append("_windsWk.nc");
    }

    if (turbOutput) {
      netCDFFileTurb = outputFileBasename;
      netCDFFileTurb.append("_turbOut.nc");
    }

    terrainOut = isSet("terrainout");
    if (terrainOut) {
      filenameTerrain = outputFileBasename;
      filenameTerrain.append("_terrainOut.obj");
    }

    // doEulDataOutput     = isSet( "doEulDataOutput" );
    // if (terrainOut) {
    //     outputEulerianFile = outputFolder + caseBaseName + "_eulerianData.nc";
    // }
    if (compPlume) {
      outputPlumeFile = outputFileBasename + "_plumeOut.nc";
      if (particleOutput) {
        outputParticleDataFile = outputFileBasename + "_particleInfo.nc";
      }
    }

  } else {
    QESout::warning("No output basename set -> output turned off");
    visuOutput = false;
    wkspOutput = false;
    turbOutput = false;
    terrainOut = false;

    plumeOutput = false;
    particleOutput = false;
  }

  plumeParameters.outputFileBasename = outputFileBasename;
  plumeParameters.plumeOutput = plumeOutput;
  plumeParameters.particleOutput = particleOutput;

  std::cout << "Summary of QES options: " << std::endl;
  std::cout << "----------------------------" << std::endl;
  // parameter files:
  std::cout << "qesWindsParamFile: " << qesWindsParamFile << std::endl;
  if (compPlume) {
    std::cout << "qesPlumeParamFile: " << qesPlumeParamFile << std::endl;
  }

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
  std::cout << "Plume model:\t\t " << (compPlume ? "ON" : "OFF") << std::endl;
  std::cout << "Verbose:\t\t " << (verbose ? "ON" : "OFF") << std::endl;

  // output files:
  if (!outputFileBasename.empty()) {
    std::cout << "----------------------------" << std::endl;
    std::cout << "Output file basename:        " << outputFileBasename << std::endl;
    std::cout << "Winds visualization output:  " << (visuOutput ? "ON" : "OFF") << std::endl;
    std::cout << "Winds workspace output:      " << (wkspOutput ? "ON" : "OFF") << std::endl;
    std::cout << "Winds terrain mesh output:   " << (terrainOut ? "ON" : "OFF") << std::endl;
    std::cout << "Turbulence output:           " << (turbOutput ? "ON" : "OFF") << std::endl;
    std::cout << "Plume output:                " << (plumeOutput ? "ON" : "OFF") << std::endl;
    std::cout << "Plume particle output:       " << (particleOutput ? "ON" : "OFF") << std::endl;
  }

  std::cout << "###################################################################" << std::endl;
}
