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

Args::Args(): inputFileUrb("cudaurb.nc"), inputFileTurb("cudaturb.nc")
{
    reg("help",          "help/usage information",                   ArgumentParsing::NONE,   '?');
    reg("quicFile",      "specifies xml settings file",              ArgumentParsing::STRING, 'q');
    reg("inputFileUrb",  "specifies input file from cuda-urb",       ArgumentParsing::STRING, 'u');
    reg("inputFileTurb", "specifies input file from cuda-turb",      ArgumentParsing::STRING, 't');
    reg("outputFileEul", "select output file for Eulerian data",     ArgumentParsing::STRING, 'o');
    reg("outputFileLag", "select output file for Lagrangian data",   ArgumentParsing::STRING, 'l');
    reg("debugOutputFolder",    "specifies folder for debug output text files", ArgumentParsing::STRING, 'd');
}

void Args::processArguments(int argc, char *argv[])
{
    processCommandLineArgs(argc, argv);

    if (isSet("help")) {
        printUsage();
        exit(EXIT_SUCCESS);
    }
    
    isSet( "quicFile", quicFile );
    isSet( "inputFileUrb", inputFileUrb );
    isSet( "inputFileTurb", inputFileTurb );    
    isSet( "outputFileEul", outputFileEul );
    isSet( "outputFileLag", outputFileLag );
    isSet( "debugOutputFolder", debugOutputFolder );
}
