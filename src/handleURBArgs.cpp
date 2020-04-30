#include "handleURBArgs.h"

URBArgs::URBArgs()
    : verbose(false),compTurb(false),
      quicFile(""), netCDFFileBasename(""),
      visuOutput(false), wkspOutput(false), turbOutput(false), terrainOut(false), 
      solveType(1), compareType(0)
{
    reg("help", "help/usage information", ArgumentParsing::NONE, '?');
    reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');
    
    reg("windsolveroff", "Turns off the wind solver and wind output", ArgumentParsing::NONE, 'x');    
    reg("solvetype", "selects the method for solving the windfield", ArgumentParsing::INT, 's');
    reg("juxtapositiontype", "selects a second solve method to compare to the original solve type", ArgumentParsing::INT, 'j');

    reg("turbcomp", "Turns on the computation of turbulent fields", ArgumentParsing::NONE, 't');

    reg("quicproj", "Specifies the QUIC Proj file", ArgumentParsing::STRING, 'q');

    reg("outbasename", "Specifies the basename for netcdf files", ArgumentParsing::STRING, 'o');
    reg("visuout", "Turns on the netcdf file to write visulatization results", ArgumentParsing::NONE, 'z');
    reg("wkout", "Turns on the netcdf file to write wroking file", ArgumentParsing::NONE, 'w');
    // [FM] the output of turbulence field linked to the flag compTurb
    //reg("turbout", "Turns on the netcdf file to write turbulence file", ArgumentParsing::NONE, 'r');   
    reg("terrainout", "Turn on the output of the triangle mesh for the terrain", ArgumentParsing::NONE, 'h');
}

void URBArgs::processArguments(int argc, char *argv[])
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

    solveWind = isSet("windsolveroff");
    if (solveWind) std::cout << "the wind fields are not being calculated" << std::endl;
    
    isSet("solvetype", solveType);
    if (solveType == CPU_Type) std::cout << "Solving with: CPU" << std::endl;
    else if (solveType == DYNAMIC_P) std::cout << "Solving with: GPU" << std::endl;

    isSet("juxtapositiontype", compareType);
    if (compareType == CPU_Type) std::cout << "Comparing against: CPU" << std::endl;
    else if (compareType == DYNAMIC_P) std::cout << "Comparing against: GPU" << std::endl;
    
    compTurb = isSet("turbcomp");
    if (compTurb) std::cout << "Turbulence model: ON" << std::endl;

    isSet( "quicproj", quicFile );
    if (quicFile != "") std::cout << "quicproj set to " << quicFile << std::endl;
        
    isSet( "outbasename", netCDFFileBasename);
    if(netCDFFileBasename != "") {
        visuOutput= isSet("visuout");
        if(visuOutput) {
            netCDFFileVisu = netCDFFileBasename;
            netCDFFileVisu.append("_windsOut.nc");
            std::cout << "Visualization NetCDF output file set to " << netCDFFileVisu << std::endl;
        }

        wkspOutput=isSet("wkout");
        if(wkspOutput) {
            netCDFFileWksp = netCDFFileBasename;
            netCDFFileWksp.append("_windsWk.nc");
            std::cout << "Workspace NetCDF output file set to " << netCDFFileWksp << std::endl;
        }
        
        // [FM] the output of turbulence field linked to the flag compTurb
        // -> subject to change 
        turbOutput=compTurb;//isSet("turbout");
        if(turbOutput) {
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
        
    } else {
        std::cout << "No output basename set -> output turned off " << std::endl;
        visuOutput=false;
        wkspOutput=false;
        turbOutput=false;
        terrainOut=false;
    }
    
}
