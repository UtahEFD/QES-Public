#include "handleQESArgs.h"

QESArgs::QESArgs()
  : verbose(false), compTurb(false), compPlume(false),
    inputWINDSFile(""), inputPlumeFile(""),
    netCDFFileBasename(""),
    visuOutput(false), wkspOutput(false), turbOutput(false), terrainOut(false), doParticleDataOutput(false),
    solveType(1), compareType(0)
{
  reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');

  reg("windsolveroff", "Turns off the wind solver and wind output", ArgumentParsing::NONE, 'x');
  reg("solvetype", "selects the method for solving the windfield", ArgumentParsing::INT, 's');
  reg("juxtapositiontype", "selects a second solve method to compare to the original solve type", ArgumentParsing::INT, 'j');

  reg("windproj", "Specifies the QES Proj file", ArgumentParsing::STRING, 'q');
  reg("plumeproj", "Specifies the QES Proj file", ArgumentParsing::STRING, 'p');

  reg("turbcomp", "Turns on the computation of turbulent fields", ArgumentParsing::NONE, 't');

  reg("outbasename", "Specifies the basename for netcdf files", ArgumentParsing::STRING, 'o');
  reg("visuout", "Turns on the netcdf file to write visulatization results", ArgumentParsing::NONE, 'z');
  reg("wkout", "Turns on the netcdf file to write wroking file", ArgumentParsing::NONE, 'w');
  // [FM] the output of turbulence field linked to the flag compTurb
  //reg("turbout", "Turns on the netcdf file to write turbulence file", ArgumentParsing::NONE, 'r');
  reg("terrainout", "Turn on the output of the triangle mesh for the terrain", ArgumentParsing::NONE, 'h');

  // going to assume concentration is always output. So these next options are like choices for additional debug output
  //reg("doEulDataOutput",     "should debug Eulerian data be output",           ArgumentParsing::NONE,   'e');
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

  verbose = isSet("verbose");
  if (verbose) std::cout << "Verbose Output: ON" << std::endl;

  isSet("windproj", inputWINDSFile);
  if (inputWINDSFile != "") std::cout << "QES proj set to " << inputWINDSFile << std::endl;

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

  isSet("juxtapositiontype", compareType);
  if (compareType == CPU_Type)
    std::cout << "Comparing against: CPU" << std::endl;
  else if (compareType == DYNAMIC_P)
    std::cout << "Comparing against: GPU" << std::endl;

  compTurb = isSet("turbcomp");
  isSet("plumeproj", inputPlumeFile);

  if (inputPlumeFile != "") {
    compTurb = true;
    std::cout << "Turbulence model: ON" << std::endl;
    compPlume = true;
    std::cout << "Plume model: ON" << std::endl;
    std::cout << "Plume file set to " << inputPlumeFile << std::endl;
  } else if (compTurb) {
    std::cout << "Turbulence model: ON" << std::endl;
  }

  isSet("outbasename", netCDFFileBasename);
  if (netCDFFileBasename != "") {
    visuOutput = isSet("visuout");
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
    turbOutput = compTurb;//isSet("turbout");
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

    //doEulDataOutput     = isSet( "doEulDataOutput" );
    //if (terrainOut) {
    //    outputEulerianFile = outputFolder + caseBaseName + "_eulerianData.nc";
    //}
    if (compPlume) {
      outputPlumeFile = netCDFFileBasename + "_conc.nc";

      doParticleDataOutput = isSet("doParticleDataOutput");

      if (doParticleDataOutput) {
        outputParticleDataFile = netCDFFileBasename + "_particleInfo.nc";
      }
    }

  } else {
    std::cout << "No output basename set -> output turned off " << std::endl;
    visuOutput = false;
    wkspOutput = false;
    turbOutput = false;
    terrainOut = false;

    //doEulDataOutput = false;
    doParticleDataOutput = false;
  }
}
