/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA-nizing QUICurb.
* Source: Adapted from init.f90, sort.f90, building_param.f90, defbuild.f90 and
*         wallbc.f90 in QUICurbv5.?
*/

#ifndef URBSETUP_H
#define URBSETUP_H

#include <iomanip>
#include <iostream>

#include "urbModule.h"

extern "C" void cudaSetupBndryMats(QUIC::boundaryMatrices d_bndrs, QUIC::celltypes d_typs);

namespace QUIC
{
	/**
	* The urbSetup class is a friend of the urbModule class that is responsible
	* initialization the proper data components i.e. initial velocities in the 
	* urbModule class so that the urbCUDA class has the needed data to run the
	* SOR iteration scheme and solve the problem described in urbModule.
	*/
	class urbSetup
	{

		public:
		
			static void setup(QUIC::urbModule* um);
		
			static void usingCPP    (QUIC::urbModule* um);
			static void usingCUDA   (QUIC::urbModule* um);
		
		protected:
		
			static void initializeBuildings(QUIC::urbModule* um);
			static void initializeSensors  (QUIC::urbModule* um);
			
			static void checkForVegetation(QUIC::urbModule* um);
			
			static void determineInitialVelocities(QUIC::urbModule* um);
			static void buildingParameterizations(QUIC::urbModule* um);
			static void streetIntersections(QUIC::urbModule* um);
			static void setupBoundaryMatrices(QUIC::urbModule* um);
		
			static void compressBoundaries
			(
				QUIC::urbModule* um,
				double* e, double* f, 
				double* g, double* h,
				double* m, double* n,
				double* o, double* p, double* q
			);
		
		private:
			
	};
}

#endif

