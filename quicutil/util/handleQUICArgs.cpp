#include "handleQUICArgs.h"

using namespace sivelab;

QUICArgs::QUICArgs()
  : verbose(false), quicproj(""), cycle(""), headless(false), fullscreen(false)
{
  reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');
  reg("quicproj", "Specifies the QUIC Proj file", ArgumentParsing::STRING, 'q');
  reg("fullscreen", "Use fullscreen mode", ArgumentParsing::NONE, 'f');
  reg("headless", "Headless mode - no graphics", ArgumentParsing::NONE, 'g');
  reg("batch", "Batch mode - no graphics/interactivity", ArgumentParsing::STRING, 'b');
  reg("surface", "Specifies the surface properties XML file", ArgumentParsing::STRING, 's');
  // KAB 09/05/14- Adding to point to input deck path
  reg("ttm_input_path", "XML file specifying TTM simulation parameters", ArgumentParsing::STRING, 't');
}

void QUICArgs::process(int argc, char *argv[])
{
  processCommandLineArgs(argc, argv);

  if (isSet("help"))
    {
      printUsage();
      exit(EXIT_SUCCESS);
    }

  verbose = isSet("verbose");
  if (verbose) std::cout << "Verbose Output: ON" << std::endl;

  isSet( "quicproj", quicproj );
  if (verbose) std::cout << "quicproj set to " << quicproj << std::endl;

  isSet( "batch", cycle );
  if (verbose) std::cout << "batch set to " << cycle << std::endl;
  
  isSet( "surface", surface);
  if (verbose) std::cout << "surface set to " << surface << std::endl;

  fullscreen = isSet("fullscreen");
  if (verbose) std::cout << "fullscreen set to " << fullscreen << std::endl;

  headless = isSet("headless");
  if (verbose) std::cout << "headless set to " << headless << std::endl;

  isSet( "ttm_input_path", ttm_input_path );
  if (verbose) std::cout << "ttm_input_path set to " << ttm_input_path << std::endl;
}

