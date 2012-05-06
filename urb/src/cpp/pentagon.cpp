#include "pentagon.h"

namespace QUIC
{
	void pentagon::initialize(float dx, float dy, float dz) {}
	
	void pentagon::upwind		(boundaryMatrices bm, velocities ivo) const {}
	void pentagon::wake			(boundaryMatrices bm, velocities ivo) const {}
	void pentagon::rooftop	(boundaryMatrices bm, velocities ivo) const {}
	void pentagon::courtyard(boundaryMatrices bm, velocities ivo) const {}
}

