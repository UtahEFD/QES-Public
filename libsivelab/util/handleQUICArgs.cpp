#include "handleQUICArgs.h"

using namespace sivelab;

QUICArgs::QUICArgs()
  : verbose(false), quicproj(""), headless(false), cycle(""), fullscreen(false)
{
}

void QUICArgs::process(int argc, char *argv[])
{
  ArgumentParsing argParser;

  argParser.reg("quicproj", "Specifies the QUIC Proj file", ArgumentParsing::STRING, 'q');
  argParser.reg("fullscreen", "Use fullscreen mode", ArgumentParsing::NONE, 'f');
  argParser.reg("headless", "Headless mode - no graphics", ArgumentParsing::NONE, 'g');
  argParser.reg("batch", "Batch mode - no graphics/interactivity", ArgumentParsing::STRING, 'b');

  argParser.processCommandLineArgs(argc, argv);

  if (argParser.isSet("help"))
    {
      argParser.printUsage();
      exit(EXIT_SUCCESS);
    }

  verbose = argParser.isSet("verbose");
  if (verbose) std::cout << "Verbose Output: ON" << std::endl;

  argParser.isSet("quicproj", quicproj);
  argParser.isSet("batch", cycle);

  fullscreen = argParser.isSet("fullscreen");
  headless = argParser.isSet("headless");
}

