/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Refactoring into C++ prior to implementation in CUDA.
* Source: Adapted from street_intersect.f90 and poisson.f90 in QUICurbv5.? 
*         (Fortran code).
*/

#ifndef INTERSECT
#define INTERSECT

#include "boundaryMatrices.h"
#include "celltypes.h"
#include "quicutil/velocities.h"

namespace QUIC
{
	class intersect
	{
		public:
		
			static void street(celltypes typs);
			
			static void poisson
			(
				velocities vels, 
				boundaryMatrices const& bndrs,
				celltypes const& typs,
				float const& dx, float const& dy, float const& dz,
				unsigned int iterations = 10
			);

		protected:
			
			static bool isCanyonBuildingOrFluidQ(CellType& celltype);
		
	};
}

#endif

