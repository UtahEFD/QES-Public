#include "handleURBArgs.h"

URBArgs::URBArgs()
	: verbose(false), quicFile(""), netCDFFile("cudaurb.nc"), cellFace(false), solveType(1), compareType(0)
{
    reg("help", "help/usage information", ArgumentParsing::NONE, '?');
    reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');
    reg("cellface", "select cellface, if not then cell center", ArgumentParsing::NONE, 'c');
    reg("solvetype", "selects the method for solving the windfield", ArgumentParsing::INT, 's');
    reg("juxtapositiontype", "selects a second solve method to compare to the original solve type", ArgumentParsing::INT, 'j');
    reg("quicproj", "Specifies the QUIC Proj file", ArgumentParsing::STRING, 'q');
    reg("netcdfout", "Specifies the netcdf file to write results to", ArgumentParsing::STRING, 'o');
    reg("demfile", "Specifies the DEM file that should be used for terrain", ArgumentParsing::STRING, 'd');
    reg("icellout", "Specifies that the iCellFlag values should be output, this also will output cutCellFlags if they exist", ArgumentParsing::NONE, 'i');
    reg("terrainout", "Specifies that the triangle mesh for the terrain should be output", ArgumentParsing::NONE, 't');
    reg("windsolveroff", "Turns off the wind solver and wind output", ArgumentParsing::NONE, 'x');
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
    
    isSet( "quicproj", quicFile );
    if (quicFile != "") std::cout << "quicproj set to " << quicFile << std::endl;

    isSet( "netcdfout", netCDFFile );
    if (netCDFFile != "") std::cout << "netCDF output file set to " << netCDFFile << std::endl;

    cellFace = isSet("cellface");
    if (cellFace) std::cout << "Cell face computations: ON" << std::endl;

    isSet("icellout", iCellOut);
    if (iCellOut != "") std::cout << "iCellFlag value output is ON" << std::endl;

    terrainOut = isSet("terrainout");
    if (terrainOut) std::cout << "the terrain triangle mesh WILL be output to terrain.obj" << std::endl;

    solveWind = isSet("windsolveroff");
    if (solveWind) std::cout << "the wind fields are not being calculated" << std::endl;

    isSet("solvetype", solveType);
    if (solveType == CPU_Type) std::cout << "Solving with: CPU" << std::endl;
    else if (solveType == DYNAMIC_P) std::cout << "Solving with: GPU" << std::endl;

    isSet("juxtapositiontype", compareType);
    if (compareType == CPU_Type) std::cout << "Comparing against: CPU" << std::endl;
    else if (compareType == DYNAMIC_P) std::cout << "Comparing against: GPU" << std::endl;

    isSet( "demfile", demFile );
    if (demFile != "") std::cout << "DEM input file set to " << demFile << std::endl;
}
