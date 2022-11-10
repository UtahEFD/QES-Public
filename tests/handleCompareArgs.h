#pragma once

/*
 * This class handles different commandline options and arguments
 * and places the values into variables. This inherits from Argument Parsing
 */

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


  bool verbose;

  // input files (from the command line)
  std::string inputWINDSFile1 = "";
  std::string inputWINDSFile2 = "";

private:
};
