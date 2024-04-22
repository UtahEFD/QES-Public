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

/** @file handlePlumeArgs.cpp */

#include "handlePlumeArgs.h"

PlumeArgs::PlumeArgs()
  : verbose(false),
    qesPlumeParamFile(""), inputTURBFile("")

{
  reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');


  reg("qesPlumeParamFile", "specifies input xml settings file", ArgumentParsing::STRING, 'q');
  // single name for all input/output files
  reg("projectQESFiles", "specifies input/output files name", ArgumentParsing::STRING, 'm');
  // individual names for input/output files
  reg("inputWINDSFile", "specifies input qes-winds file", ArgumentParsing::STRING, 'w');
  reg("inputTURBFile", "specifies input qes-turb file", ArgumentParsing::STRING, 't');
  reg("outputbasename", "Specifies the basename for netcdf files", ArgumentParsing::STRING, 'o');
  // going to assume concentration is always output. So these next options are like choices for additional debug output
  reg("plumeOutput", "should debug Lagrangian data be output", ArgumentParsing::NONE, 'l');
}

void PlumeArgs::processArguments(int argc, char *argv[])
{
  processCommandLineArgs(argc, argv);

  if (isSet("help")) {
    printUsage();
    exit(EXIT_SUCCESS);
  }

  isSet("qesPlumeParamFile", qesPlumeParamFile);
  if (qesPlumeParamFile.empty()) {
    QESout::error("qesPlumeParamFile not specified");
  }

  verbose = isSet("verbose");
  if (verbose) {
    QESout::setVerbose();
  }
  plumeOutput = isSet("plumeOutput");

  if (isSet("projectQESFiles", projectQESFiles)) {
    inputWINDSFile = projectQESFiles + "_windsWk.nc";
    inputTURBFile = projectQESFiles + "_turbOut.nc";
    outputPlumeFile = projectQESFiles + "_plumeOut.nc";
    outputParticleDataFile = projectQESFiles + "_particleInfo.nc";
  } else {
    if (!isSet("inputWINDSFile", inputWINDSFile)) {
      QESout::error("inputWINDSFile not specified!");
    }
    if (!isSet("inputTURBFile", inputTURBFile)) {
      QESout::error("inputTURBFile not specified!");
    }

    isSet("outputbasename", outputFileBasename);
    if (!outputFileBasename.empty()) {
      // setup specific output file variables for netcdf output
      outputPlumeFile = outputFileBasename + "_plumeOut.nc";
      outputParticleDataFile = outputFileBasename + "_particleInfo.nc";
    } else {
      QESout::error("No output basename set -> output turned off ");
    }
  }

  plumeParameters.outputFileBasename = outputFileBasename;
  plumeParameters.plumeOutput = plumeOutput;
  plumeParameters.particleOutput = particleOutput;

  std::cout << "Summary of QES PLUME options: " << std::endl;
  std::cout << "----------------------------" << std::endl;
  std::cout << "Plume model in stand alone mode" << std::endl;
  std::cout << "Verbose:\t\t " << (verbose ? "ON" : "OFF") << std::endl;
  std::cout << "----------------------------" << std::endl;
  std::cout << "qesPlumeParamFile set to " << qesPlumeParamFile << std::endl;
  std::cout << "----------------------------" << std::endl;
  std::cout << "Winds input file set:        " << inputWINDSFile << std::endl;
  std::cout << "Turbulence input file set:   " << inputTURBFile << std::endl;
  std::cout << "----------------------------" << std::endl;
  if (!outputFileBasename.empty()) {
    std::cout << "----------------------------" << std::endl;
    std::cout << "Output file basename:        " << outputFileBasename << std::endl;
    std::cout << "Plume output:                " << (plumeOutput ? "ON" : "OFF") << std::endl;
    std::cout << "Plume particle output:       " << (particleOutput ? "ON" : "OFF") << std::endl;
  }

  std::cout << "###################################################################" << std::endl;
}
