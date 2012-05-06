/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Group the velocity pointers together and give a nice typing to 
*         functions that take the velocities as arguments.
*					Part of QUICurbCUDA.
*/

#ifndef VELOCITYFIELD
#define VELOCITYFIELD

#include <iostream>
#include <vector_types.h>

#include "../constants.h"
#include "cellDims.h"

namespace QUIC
{
  typedef struct velocity
  {
    float u;
    float v;
    float w;
    
    operator float3() {
      float3 temp;
      temp.x = u;
      temp.y = v;
      temp.z = w;
      
      return temp;
   }
  } velocity;

	typedef struct velocityField
	{
	  velocity* vlcts;
		
		uint3 dmn;
		cellDims cll;
	} velocityField;

  inline float mag(velocity const& vlcty)
  {
    return sqrt(vlcty.u*vlcty.u + vlcty.v*vlcty.v + vlcty.w*vlcty.w);
  }
/*
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
*/
}

#endif

