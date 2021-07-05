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

/** @file Args.cpp */

#include "Args.hpp"

Args::Args() : inputWINDSFile("windsout.nc"), inputTURBFile("turbout.nc")
{
  reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  reg("inputQESFile", "specifies input xml settings file", ArgumentParsing::STRING, 'q');
  // single name for all input/output files
  reg("projectQESFiles", "specifies input/output files name", ArgumentParsing::STRING, 'm');
  // individual names for input/output files
  reg("inputWINDSFile", "specifies input qes-winds file", ArgumentParsing::STRING, 'u');
  reg("inputTURBFile", "specifies input qes-turb file", ArgumentParsing::STRING, 't');
  reg("outputFolder", "select output folder for output files", ArgumentParsing::STRING, 'o');
  reg("caseBaseName", "specify case base name for file naming", ArgumentParsing::STRING, 'b');
  // going to assume concentration is always output. So these next options are like choices for additional debug output
  reg("doEulDataOutput", "should debug Eulerian data be output", ArgumentParsing::NONE, 'e');
  reg("doParticleDataOutput", "should debug Lagrangian data be output", ArgumentParsing::NONE, 'l');
  reg("doSimInfoFileOutput", "should debug simInfoFile be output", ArgumentParsing::NONE, 's');
  // LA future work: this one should probably be replaced by cmake arguments at compiler time
  reg("debug", "should command line output include debug info", ArgumentParsing::NONE, 'd');
  reg("verbose", "should command line output include verbose info", ArgumentParsing::NONE, 'v');
}

void Args::processArguments(int argc, char *argv[])
{
  processCommandLineArgs(argc, argv);

  if (isSet("help")) {
    printUsage();
    exit(EXIT_SUCCESS);
  }


  if (!isSet("inputQESFile", inputQESFile)) {
    std::cerr << "inputQESFile not specified! Exiting program!" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (isSet("projectQESFiles", projectQESFiles)) {
    inputWINDSFile = projectQESFiles + "_windsWk.nc";
    inputTURBFile = projectQESFiles + "_turbOut.nc";
    outputFile = projectQESFiles + "_conc.nc";
    outputEulerianFile = projectQESFiles + "_eulerianData.nc";
    outputParticleDataFile = projectQESFiles + "_particleInfo.nc";
  } else {
    if (!isSet("inputWINDSFile", inputWINDSFile)) {
      std::cerr << "inputWINDSFile not specified! Exiting program!" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!isSet("inputTURBFile", inputTURBFile)) {
      std::cerr << "inputTURBFile not specified! Exiting program!" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!isSet("outputFolder", outputFolder)) {
      std::cerr << "outputFolder not specified! Exiting program!" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!isSet("caseBaseName", caseBaseName)) {
      std::cerr << "caseBaseName not specified! Exiting program!" << std::endl;
      exit(EXIT_FAILURE);
    }
    // check whether input outputFolder is an existing folder and if not, exit with error
    if (!doesDirExist(outputFolder)) {
      std::cerr << "input outputFolder \"" << outputFolder << "\" does not exist! Exiting program!" << std::endl;
      exit(EXIT_FAILURE);
    }

    // check whether input outputFolder has a "/" char on the end, and if not, add one
    std::string lastChar = outputFolder.substr(outputFolder.length() - 1, 1);
    if (lastChar != "/") {
      outputFolder = outputFolder + "/";
    }

    // LA future work: might be important to also check caseBaseName for bad characters that would make a filename have trouble
    //  I was checking to see if it is an empty string, but I think the isSet() function probably handles that check


    // now that the outputFolder and caseBaseName are confirmed to be good, setup specific output file variables for netcdf output
    outputEulerianFile = outputFolder + caseBaseName + "_eulerianData.nc";
    outputFile = outputFolder + caseBaseName + "_conc.nc";
    outputParticleDataFile = outputFolder + caseBaseName + "_particleInfo.nc";
  }

  doEulDataOutput = isSet("doEulDataOutput");
  doParticleDataOutput = isSet("doParticleDataOutput");
  doSimInfoFileOutput = isSet("doSimInfoFileOutput");
  debug = isSet("debug");
  verbose = isSet("verbose");
}
