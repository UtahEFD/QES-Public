/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Porting QUICurb to C++.
* Remark: This class is nowhere near finished. Just started and that's it.
*/

#ifndef BUILDING_PENTAGON
#define BUILDING_PENTAGON

#include "building.h"

namespace QUIC
{
	typedef struct pentagon : public building
	{	
		pentagon() : type(PENTAGON) {}
	
		const Type type;
	
		void initialize(float dx, float dy, float dz);
	
		void upwind		(boundaryMatrices bm, velocities ivo) const;
		void wake			(boundaryMatrices bm, velocities ivo) const;
		void rooftop	(boundaryMatrices bm, velocities ivo) const;
		void courtyard(boundaryMatrices bm, velocities ivo) const;
		
	} pentagon;
}

#endif	

