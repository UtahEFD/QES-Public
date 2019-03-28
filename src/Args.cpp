//
//  Args.cpp
//  
//  This class handles different commandline options and arguments 
//  and places the values into variables. 
//
//  This inherits from Argument Parsing.
//
//  Created by Pete Willemson on 10/6/09
//

#include "Args.hpp"

Args::Args(): inputFileUrb("cudaurb.nc"), inputFileTurb("cudaturb.nc"), outputFile("cudaplume.nc")
{
    reg("help",          "help/usage information",                   ArgumentParsing::NONE,   '?');
    reg("inputFileUrb",  "specifies input file from cuda-urb",       ArgumentParsing::STRING, 'u');
    reg("inputFileTurb", "specifies input file from cuda-turb",      ArgumentParsing::STRING, 't');
    reg("outputFile",    "select cellface, if not then cell center", ArgumentParsing::STRING, 'o');
}

void Args::processArguments(int argc, char *argv[])
{
    processCommandLineArgs(argc, argv);

    if (isSet("help")) {
        printUsage();
        exit(EXIT_SUCCESS);
    }
    
    isSet( "inputFileUrb", inputFileUrb );
    isSet( "inputFileTurb", inputFileTurb );    
    isSet( "outputFile", outputFile );
}
