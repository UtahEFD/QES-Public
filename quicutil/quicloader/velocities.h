/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Group the velocity pointers together and give a nice typing to 
*         functions that take the velocities as arguments.
*					Part of QUICurbCUDA.
*/

#ifndef VELOCITIES
#define VELOCITIES

//#ifdef __APPLE__
// Currently needed to get OS X 10.6+ and CUDA to play nice together.
// Supposedly, this will be fixed in Cuda 3.0
// http://forums.nvidia.com/index.php?showtopic=106592
//#undef _GLIBCXX_ATOMIC_BUILTINS
//#endif

#include <iostream>
#include <vector_types.h>

namespace QUIC
{
  static const float MAX_VELOCITY_MAGNITUDE = 100.;

  typedef struct velocities
  {
	  float* u;
	  float* v;
	  float* w;
	
	  int3 dim;
  } velocities;

  // The check that happens so often during building parameterizations.
  inline bool checkVelocityMagnitude
  (
	  std::string const& who, std::string const& where, 
	  float const& value, int const& ndx
  )
  {
	  if(value > MAX_VELOCITY_MAGNITUDE)
	  {
		  std::cerr <<
			  "Parameterized " << 
			  who << " " <<
			  "exceeds max in " << 
			  where << ". Velocity is " << 
			  value << " at " <<
			  ndx << ". Maximum allowed is " <<
			  MAX_VELOCITY_MAGNITUDE << "." <<
		  std::endl;
		
		  return true;
	  }
	  else
	  {
		  return false;
	  }
  }
}

#endif

