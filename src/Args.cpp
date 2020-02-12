//
//  Args.cpp
//  
//  This class handles different commandline options and arguments 
//  and places the values into variables. 
//
//  This inherits from Argument Parsing.
//
//  Created by Pete Willemson on 10/6/09
//  Modified by Loren Atwood 02/03/2020
//

#include "Args.hpp"

Args::Args(): inputUrbFile("cudaurb.nc"), inputTurbFile("cudaturb.nc")
{
    reg("help",                "help/usage information",                         ArgumentParsing::NONE,   '?');
    reg("inputQuicFile",       "specifies input xml settings file",              ArgumentParsing::STRING, 'q');
    reg("inputUrbFile",        "specifies input cuda-urb file",                  ArgumentParsing::STRING, 'u');
    reg("inputTurbFile",       "specifies input cuda-turb file",                 ArgumentParsing::STRING, 't');
    reg("outputFolder",        "select output folder for output files",          ArgumentParsing::STRING, 'o');
    reg("caseBaseName",        "specify case base name for file naming",         ArgumentParsing::STRING, 'b');
    // going to assume concentration is always output. So these next options are like choices for additional debug output
    reg("doEulDataOutput",     "should debug Eulerian data be output",           ArgumentParsing::NONE,   'e');
    reg("doLagrDataOutput",    "should debug Lagrangian data be output",         ArgumentParsing::NONE,   'l');
    reg("doSimInfoFileOutput", "should debug simInfoFile be output",             ArgumentParsing::NONE,   's');
    // LA future work: this one should probably be replaced by cmake arguments at compiler time
    reg("debug",               "should command line output include debug info",  ArgumentParsing::NONE,   'd');
}

void Args::processArguments(int argc, char *argv[])
{
    processCommandLineArgs(argc, argv);

    if( isSet("help") )
    {
        printUsage();
        exit(EXIT_SUCCESS);
    }
    
    if( !isSet( "inputQuicFile", inputQuicFile ) )
    {
        std::cerr << "inputQuicFile not specified! Exiting program!" << std::endl;
        exit(EXIT_FAILURE);
    }
    if( !isSet( "inputUrbFile",  inputUrbFile ) )
    {
        std::cerr << "inputUrbFile not specified! Exiting program!" << std::endl;
        exit(EXIT_FAILURE);
    }
    if( !isSet( "inputTurbFile", inputTurbFile ) )
    {
        std::cerr << "inputTurbFile not specified! Exiting program!" << std::endl;
        exit(EXIT_FAILURE);
    }  
    if( !isSet( "outputFolder",  outputFolder ) )
    {
        std::cerr << "outputFolder not specified! Exiting program!" << std::endl;
        exit(EXIT_FAILURE);
    }
    if( !isSet( "caseBaseName",  caseBaseName ) )
    {
        std::cerr << "caseBaseName not specified! Exiting program!" << std::endl;
        exit(EXIT_FAILURE);
    }
    doEulDataOutput     = isSet( "doEulDataOutput" );
    doLagrDataOutput    = isSet( "doLagrDataOutput" );
    doSimInfoFileOutput = isSet( "doSimInfoFileOutput" );
    debug               = isSet( "debug" );
    

    // check whether input outputFolder is an existing folder and if not, exit with error
    if( !doesDirExist(outputFolder) )
    {
        std::cerr << "input outputFolder \"" << outputFolder << "\" does not exist! Exiting program!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // check whether input outputFolder has a "/" char on the end, and if not, add one
    std::string lastChar = outputFolder.substr(outputFolder.length()-1,1);
    if( lastChar != "/" )
    {
        outputFolder = outputFolder + "/";
    }

    // LA future work: might be important to also check caseBaseName for bad characters that would make a filename have trouble
    //  I was checking to see if it is an empty string, but I think the isSet() function probably handles that check


    // now that the outputFolder and caseBaseName are confirmed to be good, setup specific output file variables for netcdf output
    outputEulerianFile = outputFolder + caseBaseName + "_eulerianData.nc";
    outputLagrToEulFile = outputFolder + caseBaseName + "_conc.nc";
    outputLagrangianFile = outputFolder + caseBaseName + "_particleInfo.nc";

}
