/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Refactoring QUICurb into C++ prior to a move to CUDA.
* Source: Mainly adapted from canyon.f90, rooftop.f90, rectanglewake.f90 and 
*         upwind.f90 in QUICurbv5.? (Fortran code).
*/

#ifndef URBBUILDING
#define URBBUILDING

#include <vector>

#include "quicloader/QUBuildings.h"
#include "quicloader/velocities.h"
#include "util/angle.h"

#include "../util/index3D.h"
#include "celltypes.h"

// \\todo May want to think about generating classes for upwind, wake and 
// rooftop. Do they each have their own basic parameters?
// Than a building has a instance of each described using the buildings 
// parameters. May or may not be a good idea.

namespace QUIC
{
	class urbBuilding : public quBuildings::buildingData
	{	
		public:
			urbBuilding();
			urbBuilding(quBuildings::buildingData const& thr);
			virtual ~urbBuilding() {}

      urbBuilding const& operator=(quBuildings::buildingData const& thr);

      float zo;

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
				std::vector<urbBuilding> const& bldngs,
				float const& dx, float const& dy,	float const& dz
			) const;
			
			inline virtual float getWeff(){return Weff;};
			inline virtual float getLeff(){return Leff;};

			inline virtual float getLf(){return Lf;}
                        inline virtual float getLr(){return Lr;};
		protected:
		
		  // Careful there's a gamma in BuildingData.
			float sin_gamma;
			float cos_gamma;
		
			// parameterization parameters
			sivelab::angle phiprime; // ENG
			float Weff;
			float Leff;
                        float Lr, Lf;
		
			sivelab::angle upwind_dir; // ENG
			float uo_h;
			float vo_h;
			
			sivelab::angle upwind_rel;
			float sin_upwind_dir;
			float cos_upwind_dir;
			
			float half_length; // x
			float half_width; // y
	
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
				sivelab::angle const& angle,
				float& x_u, float& x_v, float& x_w,
				float& y_u, float& y_v, float& y_w
			) const;
			
			// Uses the calling buildings info for finding along and cross canyon
			// directions.
			void determineCanyonDirections
			(
				int const& i, int const& j, float const& dx, float const& dy,
				sivelab::angle const& canyon_dir, sivelab::angle& along_dir, sivelab::angle& cross_dir, 
				bool& reverse_flag
			) const;
			
			void searchUpwindForRooftopDisruptor
			(
				celltypes ct, 
				float const& dx, float const& dy
			);
			
			sivelab::angle calculateRoofAngle(sivelab::angle const& a1, sivelab::angle const& a2) const;
		
	};

	void sortBuildings(std::vector<urbBuilding*>);
	void swapBuildings(urbBuilding* b1, urbBuilding* b2);
	
	// Given i, j and k, finds the building at that index in the domain.
	// Returns NULL if no building found.
	urbBuilding* findBuilding
	(
		std::vector<urbBuilding> bldngs, 
		int const& i, int const& j, int const& k
	);
}

#endif

