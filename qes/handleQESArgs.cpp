#include "handleQESArgs.h"

QESArgs::QESArgs()
  : verbose(false),
    qesWindsParamFile(""), qesPlumeParamFile(""),
    netCDFFileBasename(""),
    solveWind(false), compTurb(false), compPlume(false),
    solveType(1),
    visuOutput(true), wkspOutput(false), terrainOut(false),
    turbOutput(false),
    doParticleDataOutput(false)

{
  reg("help", "help/usage information", ArgumentParsing::NONE, '?');
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
  reg("terrainout", "Turn on the output of the triangle mesh for the terrain", ArgumentParsing::NONE, 'h');

  // going to assume concentration is always output. So these next options are like choices for additional debug output
  // reg("doEulDataOutput",     "should debug Eulerian data be output",           ArgumentParsing::NONE,   'e');
  reg("doParticleDataOutput", "should debug Lagrangian data be output", ArgumentParsing::NONE, 'l');
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
  }

  verbose = isSet("verbose");
  if (verbose) {
    QESout::setVerbose();
  }

  std::cout << "Summary of QES options: " << std::endl;
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

  if (compPlume) {
    std::cout << "Plume model:\t\t ON" << std::endl;
  } else {
    std::cout << "Plume model:\t\t OFF" << std::endl;
  }

  if (verbose) {
    std::cout << "Verbose:\t\t ON" << std::endl;
  } else {
    std::cout << "Verbose:\t\t OFF" << std::endl;
  }

  std::cout << "----------------------------" << std::endl;

  std::cout << "qesWindsParamFile set to " << qesWindsParamFile << std::endl;
  if (compPlume) {
    std::cout << "qesPlumeParamFile set to " << qesPlumeParamFile << std::endl;
  }

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

    // doEulDataOutput     = isSet( "doEulDataOutput" );
    // if (terrainOut) {
    //     outputEulerianFile = outputFolder + caseBaseName + "_eulerianData.nc";
    // }
    if (compPlume) {
      outputPlumeFile = netCDFFileBasename + "_plumeOut.nc";

      doParticleDataOutput = isSet("doParticleDataOutput");

      if (doParticleDataOutput) {
        outputParticleDataFile = netCDFFileBasename + "_particleInfo.nc";
      }
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
    if (!outputPlumeFile.empty()) {
      std::cout << "[PLUME]\t Plume NetCDF output file set:\t\t " << outputPlumeFile << std::endl;
    }
    if (!outputParticleDataFile.empty()) {
      std::cout << "[PLUME]\t Particle NetCDF output file set:\t " << outputParticleDataFile << std::endl;
    }

  } else {

    QESout::warning("No output basename set -> output turned off");
    visuOutput = false;
    wkspOutput = false;
    turbOutput = false;
    terrainOut = false;

    // doEulDataOutput = false;
    doParticleDataOutput = false;
  }

  std::cout << "-------------------------------------------------------------------" << std::endl;
}
