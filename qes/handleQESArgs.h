#pragma once

/*
 * This class handles different commandline options and arguments
 * and places the values into variables. This inherits from Argument Parsing
 */

#include <iostream>
#include "util/ArgumentParsing.h"
#include "util/QESout.h"

enum solverTypes : int { CPU_Type = 1,
                         DYNAMIC_P = 2,
                         Global_M = 3,
                         Shared_M = 4 };

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
  // going to assume concentration is always output. So these next options are like choices for additional debug output
  bool plumeOutput, particleOutput;

  // output file variables created from the outputFolder and caseBaseName

  // std::string outputEulerianFile;
  std::string outputPlumeFile;
  std::string outputParticleDataFile;

private:
};
