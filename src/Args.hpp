//
//  Args.hpp
//  
//  This class handles different commandline options and arguments 
//  and places the values into variables. 
//
//  This inherits from Argument Parsing.
//
//  Created by Pete Willemson on 10/6/09
//  Modified by Loren Atwood on 02/03/2020
//

#ifndef ARGS_HPP
#define ARGS_HPP


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
        std::string inputWINDSFile = "";
        std::string inputTURBFile = "";
        std::string outputFolder = "";
        std::string caseBaseName = "";
        // going to assume concentration is always output. So these next options are like choices for additional debug output
        bool doEulDataOutput;
        bool doLagrDataOutput;
        bool doSimInfoFileOutput;
        // LA future work: this one should probably be replaced by cmake arguments at compiler time
        bool debug;


        // output file variables created from the outputFolder and caseBaseName
        std::string outputEulerianFile;
        std::string outputLagrToEulFile;
        std::string outputLagrangianFile;

    private:
};

#endif
