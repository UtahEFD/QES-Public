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

/** @file Args.hpp 
 * @brief This class handles different commandline options and arguments
 * and places the values into variables.
 *
 * @note Child of ArgumentParsing
 * @sa ArgumentParsing
 */

#pragma once

#include <iostream>

#include "util/doesFolderExist.h"
#include "util/ArgumentParsing.h"

class Args : public ArgumentParsing
{
public:
  Args();

  ~Args()
  {
  }

  /*
     * Takes in the commandline arguments and places
     * them into variables.
     *
     * @param argc -number of commandline options/arguments
     * @param argv -array of strings for arguments
     */
  void processArguments(int argc, char *argv[]);

  std::string inputQESFile = "";
  std::string projectQESFiles = "";
  std::string inputWINDSFile = "";
  std::string inputTURBFile = "";
  std::string outputFolder = "";
  std::string caseBaseName = "";
  // going to assume concentration is always output. So these next options are like choices for additional debug output
  bool doEulDataOutput;
  bool doParticleDataOutput;
  bool doSimInfoFileOutput;
  // LA future work: this one should probably be replaced by cmake arguments at compiler time
  bool debug = false;
  bool verbose = false;

  // output file variables created from the outputFolder and caseBaseName
  std::string outputEulerianFile;
  std::string outputFile;
  std::string outputParticleDataFile;

private:
};
