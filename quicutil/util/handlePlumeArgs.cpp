#include "handlePlumeArgs.h"

using namespace sivelab;

PlumeArgs::PlumeArgs()
  : verbose(false), quicFile(""), numParticles(0),
    concFile(""), concId(0), fullscreen(false), 
    networkMode(-1), viewingMode(0), headless(false)
{
  reg("help", "help/usage information", ArgumentParsing::NONE, 'h');
  reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');

  reg("version", "Print version information", ArgumentParsing::NONE);

  reg("quicproj", "Specifies the QUIC Proj file", ArgumentParsing::STRING, 'q');

  reg("numParticles", "Number of particles to use in simulation", ArgumentParsing::INT, 'p');
  reg("concFile", "Concentration output file name", ArgumentParsing::STRING, 'c');
  reg("concId", "Integer used to identify concentration data", ArgumentParsing::INT, 'i');


  reg("networkmode", "Use network mode (broadcast for Treadport)", ArgumentParsing::INT, 'n');
  reg("viewingmode", "Which viewing mode to use (Treadport mode)", ArgumentParsing::INT, 'm');
  reg("treadportview", "Which Treadport View", ArgumentParsing::STRING, 't');
  reg("dynamicTreadportFrustum", "Use dynamic frustum code.", ArgumentParsing::NONE, 'd');

  reg("offscreenRender", "Use the off-screen rendering system", ArgumentParsing::NONE, 'r');
  reg("ignoreSignal", "Whether to ignore the kill SIGNAL", ArgumentParsing::NONE, 's');

  reg("headless", "Headless mode (identical to offscreenRender option", ArgumentParsing::NONE, 'g');
}

void PlumeArgs::process(int argc, char *argv[])
{


	std::cout<<"in process of handlePlume ards"<<std::endl;

  processCommandLineArgs(argc, argv);
std::cout<<"--------------------------------------------- done with procesCommandLIne argas"<<std::endl;
  if (isSet("help"))
    {
      printUsage();
      exit(EXIT_SUCCESS);
    }

  verbose = isSet("verbose");
  if (verbose) std::cout << "Verbose Output: ON" << std::endl;

  int nParts = 0;
  isSet("numParticles", nParts);
  numParticles = nParts;
  std::cout << "Num Parts = " << numParticles << std::endl;
  
  isSet("quicproj", quicFile);

  isSet("concFile", concFile);



  std::cout<<"Analysis of the concID"<<std::endl;
  int cId = 0;
  isSet("concId", cId);
  concId = cId;

  fullscreen = isSet("fullscreen");

  isSet("networkmode", networkMode);

  isSet("viewingmode", viewingMode); 

  isSet("treadportview", treadportView);
  // treadportView = argVal.c_str()[0];  <<-- CHAR?? check on this -Pete

  // if (isSet("dynamicTreadportFrustum"))
    // utilPtr->static_treadport_frustum = 0;

  ignoreSignal = isSet("ignoreSignal");
  
  offscreenRender = isSet("offscreenRender");

  headless = isSet("headless");
}

