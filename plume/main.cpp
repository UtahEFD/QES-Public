#include <iostream>
#include <cstdlib>
#include <string>

#include "util/handleQUICArgs.h"
#include "quicutil/QUICProject.h"

sivelab::QUICProject *data;

int main(int argc, char** argv) 
{
  // Argument parsing is handled via the utility libraries in the
  // libsivelab codebase.
  sivelab::QUICArgs quicArgs;
  quicArgs.process( argc, argv );

  // Create a new QUIC data object using the libsivelab codebase for
  // loading QUIC data.
  // 
  // To run the code and provide the quicproj file, you can do this:
  //
  // ./plume -q <REPLACE WITH YOUR PATH>/quicdata/SBUE_small_bldg/SBUE_small_bldg.proj 

  data = new sivelab::QUICProject( quicArgs.quicproj );
  std::cout << "Done loading QUIC data.\n" << std::endl;

  // You could work to load the data in QUICProject into CUDA memory...
  // 1. Wind velocities - domain data
  //       While you wait to load actual wind data... why not just create random wind data???  drand48()
  // 2. Find number of particles in qpParamData structure of QUICProject, and then create CUDA memory to 
  //    hold the particles...
  //
  // 3.  Build kernel to simply advect the particles...
  //
  // 4. Use time step in QPParams.h to determine the
  // loop... while (simulation duration is not yet complete) run
  // advection kernel again...

  delete data;
}
