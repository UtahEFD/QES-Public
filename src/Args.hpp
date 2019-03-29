//
//  Args.hpp
//  
//  This class handles different commandline options and arguments 
//  and places the values into variables. 
//
//  This inherits from Argument Parsing.
//
//  Created by Pete Willemson on 10/6/09
//

#ifndef ARGS_HPP
#define ARGS_HPP

#include <iostream>
#include "util/ArgumentParsing.h"

class Args : public ArgumentParsing 
{
    public:
    
        Args();
    
        ~Args() {}
    
        /*
         * Takes in the commandline arguments and places
         * them into variables.
         *
         * @param argc -number of commandline options/arguments
         * @param argv -array of strings for arguments
         */
        void processArguments(int argc, char *argv[]);
        
        std::string quicFile = "";
        std::string inputFileUrb = "";
        std::string inputFileTurb = "";
        std::string outputFile = "";
    
    private:
};

#endif