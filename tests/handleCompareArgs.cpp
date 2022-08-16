#include "handleCompareArgs.h"

Args::Args()
  : verbose(false),
    inputWINDSFile1(""), inputWINDSFile2("")
{
  reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');

  reg("windsfile1", "Specifies the WINDS file 1", ArgumentParsing::STRING, 'a');
  reg("windsfile2", "Specifies the WINDS file 2", ArgumentParsing::STRING, 'b');
}


void Args::processArguments(int argc, char *argv[])
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

  isSet("windsfile1", inputWINDSFile1);
  if (inputWINDSFile1 != "") {
    std::cout << "Specifies the WINDS file 1: " << inputWINDSFile1 << std::endl;
  } else {
    std::cout << "[ERROR] WINDS file 1 missing " << std::endl;
    exit(EXIT_FAILURE);
  }

  isSet("windsfile2", inputWINDSFile2);
  if (inputWINDSFile2 != "") {
    std::cout << "Specifies the WINDS file 2: " << inputWINDSFile2 << std::endl;
  } else {
    std::cout << "[ERROR] WINDS file 2 missing " << std::endl;
    exit(EXIT_FAILURE);
  }
}
