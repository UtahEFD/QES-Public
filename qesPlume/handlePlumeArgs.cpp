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

#include "handlePlumeArgs.hpp"

PlumeArgs::PlumeArgs()
  : verbose(false),
    qesPlumeParamFile(""), inputTURBFile("")

{
  reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');
  // LA future work: this one should probably be replaced by cmake arguments at compiler time
  // reg("debug", "should command line output include debug info", ArgumentParsing::NONE, 'd');

  reg("qesPlumeParamFile", "specifies input xml settings file", ArgumentParsing::STRING, 'q');
  // single name for all input/output files
  reg("projectQESFiles", "specifies input/output files name", ArgumentParsing::STRING, 'm');
  // individual names for input/output files
  reg("inputWINDSFile", "specifies input qes-winds file", ArgumentParsing::STRING, 'w');
  reg("inputTURBFile", "specifies input qes-turb file", ArgumentParsing::STRING, 't');
  reg("outbasename", "Specifies the basename for netcdf files", ArgumentParsing::STRING, 'o');
  // going to assume concentration is always output. So these next options are like choices for additional debug output
  reg("doEulDataOutput", "should debug Eulerian data be output", ArgumentParsing::NONE, 'e');
  reg("doParticleDataOutput", "should debug Lagrangian data be output", ArgumentParsing::NONE, 'l');
  reg("doSimInfoFileOutput", "should debug simInfoFile be output", ArgumentParsing::NONE, 's');
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
  doEulDataOutput = isSet("doEulDataOutput");
  doParticleDataOutput = isSet("doParticleDataOutput");
  doSimInfoFileOutput = isSet("doSimInfoFileOutput");

  if (isSet("projectQESFiles", projectQESFiles)) {
    inputWINDSFile = projectQESFiles + "_windsWk.nc";
    inputTURBFile = projectQESFiles + "_turbOut.nc";
    outputFile = projectQESFiles + "_plumeOut.nc";
    outputEulerianFile = projectQESFiles + "_eulerianData.nc";
    outputParticleDataFile = projectQESFiles + "_particleInfo.nc";
  } else {
    if (!isSet("inputWINDSFile", inputWINDSFile)) {
      QESout::error("inputWINDSFile not specified!");
    }
    if (!isSet("inputTURBFile", inputTURBFile)) {
      QESout::error("inputTURBFile not specified!");
    }

    isSet("outbasename", netCDFFileBasename);
    if (!netCDFFileBasename.empty()) {
      // setup specific output file variables for netcdf output
      outputEulerianFile = netCDFFileBasename + "_eulerianData.nc";
      outputFile = netCDFFileBasename + "_plumeOut.nc";
      outputParticleDataFile = netCDFFileBasename + "_particleInfo.nc";
    } else {
      QESout::error("No output basename set -> output turned off ");
    }
  }

  plumeParameters.outputFileBasename = netCDFFileBasename;
  plumeParameters.plumeOutput = true;
  plumeParameters.particleOutput = doParticleDataOutput;

  std::cout << "Summary of QES PLUME options: " << std::endl;
  std::cout << "----------------------------" << std::endl;
  std::cout << "Plume model in stand alone mode" << std::endl;
  if (verbose) {
    std::cout << "Verbose:\t\t ON" << std::endl;
  } else {
    std::cout << "Verbose:\t\t OFF" << std::endl;
  }
  std::cout << "----------------------------" << std::endl;
  std::cout << "qesPlumeParamFile set to " << qesPlumeParamFile << std::endl;
  std::cout << "----------------------------" << std::endl;
  std::cout << "[INPUT]\t Winds NetCDF input file set:\t\t " << inputWINDSFile << std::endl;
  std::cout << "[INPUT]\t Turbulence NetCDF input file set:\t " << inputTURBFile << std::endl;
  std::cout << "----------------------------" << std::endl;
  std::cout << "[PLUME]\t Plume NetCDF output file set:\t\t " << outputFile << std::endl;
  if (doEulDataOutput) {
    std::cout << "[PLUME]\t Eulerian NetCDF output file set:\t " << outputEulerianFile << std::endl;
  }
  if (doEulDataOutput) {
    std::cout << "[PLUME]\t Particle NetCDF output file set:\t " << outputParticleDataFile << std::endl;
  }
  std::cout << "###################################################################" << std::endl;
}
