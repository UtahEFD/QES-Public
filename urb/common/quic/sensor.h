/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Porting QUICurb to C++.
* Remark: Porting to CUDA should be fairly painless.
*/

#ifndef SENSOR_SITE
#define SENSOR_SITE

#include <vector>

#include "velocities.h"

#include "../util/angle.h"
#include "../util/bisect.h"
#include "../util/constants.h"


namespace QUIC
{
	// \\ todo break this up further into a series of sites.
	// rename this sensorgrid or something. It should have
	// a series of sensor sites as a datamember, each of which
	// does things that this does now.

	class sensor
	{	
		public:
	
			enum BOUNDARY {LOG = 1, EXP = 2, URBAN_CANOPY = 3, DISCRETE = 4};

			sensor();
			~sensor();

			std::string name;
			std::string file;
		
			int x;
			int y;
		
			float time;
		
			BOUNDARY boundary;
		
			//float zo; // ? site_pp exp or zo OR exp / zo
			float pp; // aka site zo building roughness??	// extend for more time steps
			float rL;																			// extend for more time steps
			float H;																			// extend for more time steps
			float ac;																			// extend for more time steps

			int time_steps;
		
			// extend for more time steps
			float wnd_hght;		
			float wnd_spd;		
			angle wnd_drctn; // MET
		
			unsigned prfl_lgth;
		
			float* u_prof;
			float* v_prof;
		
			void print() const;
		
			void determineVerticalProfiles(float const& zo, float const& dz, int const& i_time = 0);
		
		private:
			
			static float calculatePSI_M(float const& value, bool const& basic);

	};
}

#endif

