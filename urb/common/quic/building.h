/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Refactoring QUICurb into C++ prior to a move to CUDA.
* Source: Mainly adapted from canyon.f90, rooftop.f90, rectanglewake.f90 and 
*         upwind.f90 in QUICurbv5.? (Fortran code).
*/

#ifndef BUILDING
#define BUILDING

#include <iostream>
#include <cmath>
#include <string>
#include <vector>

#include "../util/angle.h"
#include "../util/index3D.h"
#include "../util/constants.h"
#include "../util/minmax.h"

#include "boundaryMatrices.h"
#include "celltypes.h"
#include "velocities.h"


// \\todo May want to think about generating classes for upwind, wake and 
// rooftop. Do they each have their own basic parameters?
// Than a building has a instance of each described using the buildings 
// parameters. May or may not be a good idea.

namespace QUIC
{
	class building
	{	
		public:
		
			enum Type 
			{
				REGULAR    = 1, 
				CYLINDRICAL = 2, 
				PENTAGON   = 3, 
				VEGETATION = 9
			};
	
			building();
			virtual ~building() {}

			/* BEGIN from input file. */
			int num;
			int group;

			Type type; // Assume everythings a rectangle, unless told otherwise.
	  
			int lgth; // x
			int wdth; // y
			int hght; // z

			int xfo;
			int yfo;
			int zfo;
	
			float zo;
	
			angle gamma; // ENG
			int attenuation;
			/* END from input file. */
	
			// flags
			bool doUpwind;
			bool doRooftop;
			bool doWake;
		
			// types
			int upwindtype;
			int rooftype;
			int waketype;
			
			//additional data needed in nonLocalMixing
			float lf, lr, atten;
			float Sx, Sy;
			float damage;
			
			virtual void initialize
			(
				velocities const& ntlvels, 
				float const& dx, float const& dy, float const& dz
			);
	
			// Must be called after ALL buildings are initialized.
			//virtual void findDownwindBuildings(std::vector<building*> bldngs);
	
			virtual void print() const;
	
			virtual bool inBuildingQ(int const& i, int const& j, int const& k) const;
			virtual bool inBuildingQ(index3D const& loc) const;
			//virtual bool inDownwindQ(int const& i, int const& j, int const& k) const;
	
			virtual void interior
			(
				celltypes ct, velocities initial, 
				float const& dx, float const& dy, float const& dz
			) const;
	
			virtual void upwind
			(
				celltypes ct, velocities initial,
				float const& dx, float const& dy, float const& dz
			);
	
			virtual void rooftop
			(
				celltypes ct, velocities initial,
				float const& dx, float const& dy, float const& dz
			);		
	
			virtual void wake
			(
				celltypes ct, velocities initial,
				float const& dx, float const& dy, float const& dz
			);
			
			virtual void canyon
			(
				celltypes ct, velocities const& initial, 
				std::vector<building*> const& bldngs,
				float const& dx, float const& dy,	float const& dz
			) const;
			
			inline virtual float getWeff(){return Weff;};
			inline virtual float getLeff(){return Leff;};
		protected:
		
			float sin_gamma;
			float cos_gamma;
		
			// parameterization parameters
			angle phiprime; // ENG
			float Weff;
			float Leff;
			float Lr;
		
			angle upwind_dir; // ENG
			float uo_h;
			float vo_h;
			
			angle upwind_rel;
			float sin_upwind_dir;
			float cos_upwind_dir;
			
			float half_lgth; // x
			float half_wdth; // y
	
			float xco;
			float yco;
			
			int istart, iend;
			int jstart, jend;
			int kstart, kend;
			
			// Couldn't these be specified with a point.
			float x1, x2, x3, x4;
			float y1, y2, y3, y4;
			
			float xf1, xf2;
			float yf1, yf2;
			
			float xw1, xw2, xw3;
			float yw1, yw2, yw3;
			
			

			void calculateDimensionalIndices
			(
				float const& dx, 
				float const& dy, 
				float const& dz
			);
			
			void calculateFootprintCorners();
			
			void calculatePhiPrime
			(
				velocities const& initial, 
				float const& dx, float const& dy, float const& dz
			);
			
			void calculateUpwindDirection
			(
				velocities const& initial, 
				float const& dx, float const& dy, float const& dz
			);
			
			void calculateUpwindCorners();
			void calculateWakeCorners();
			
			void calculateWeffLeffLr();
			void calculateLr(float const& height);
			
			bool inSubdomainQ() const;		
			
			// \\todo Should make the following static.
			void determineVelocityLocations
			(
				int const& i, int const& j, 
				float const& dx, float const& dy,
				angle const& angle,
				float& x_u, float& x_v, float& x_w,
				float& y_u, float& y_v, float& y_w
			) const;
			
			// Uses the calling buildings info for finding along and cross canyon
			// directions.
			void determineCanyonDirections
			(
				int const& i, int const& j, float const& dx, float const& dy,
				angle const& canyon_dir, angle& along_dir, angle& cross_dir, 
				bool& reverse_flag
			) const;
			
			void searchUpwindForRooftopDisruptor
			(
				celltypes ct, 
				float const& dx, float const& dy
			);
			
			angle calculateRoofAngle(angle const& a1, angle const& a2) const;
		
	};

	void sortBuildings(std::vector<building*>);
	void swapBuildings(building* b1, building* b2);
	
	// Given i, j and k, finds the building at that index in the domain.
	// Returns NULL if no building found.
	building* findBuilding
	(
		std::vector<building*> bldngs, 
		int const& i, int const& j, int const& k
	);
}

#endif

