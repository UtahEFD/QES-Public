#include "building.h"

#include <iostream>
#include <cmath>

#include "quicutil/constants.h"
#include "../util/index3D.h"
#include "../util/minmax.h"

#include "boundaryMatrices.h"

using namespace sivelab;

namespace QUIC
{
	urbBuilding::urbBuilding()
	: quBuildings::buildingData()
	{
		istart = iend = jstart = jend = kstart = kend = 0;

		half_length = half_width = 0.;

		xco = yco = 0.;
	
		// parameterization parameters
		phiprime = upwind_dir = upwind_rel = angle(); // ENG
		Weff = Leff = Lr = 0.;		
		uo_h = vo_h = 0.;
	
		// flags
		doWake = doRooftop = doUpwind = false;
		upwindtype = rooftype = waketype = 0;
		
		// helpers
		sin_upwind_dir = cos_upwind_dir = 0.;
		
		x1 = x2 = x3 = x4 = 0.;
		y1 = y2 = y3 = y4 = 0.;
		
		xf1 = xf2 = 0.;
		yf1 = yf2 = 0.;
		
		xw1 = xw2 = xw3 = 0.;
		yw1 = yw2 = yw3 = 0.;
	}

  urbBuilding::urbBuilding(quBuildings::buildingData const& thr)
  : quBuildings::buildingData(thr)
  {}

  urbBuilding const& urbBuilding::operator=(quBuildings::buildingData const& thr)
  {
    if (this != &thr)
    {
      quBuildings::buildingData::operator=(thr);
		}
		return *this;
  }

		
	void urbBuilding::initialize
	(
		velocities const& ntls, 
		float const& dx, float const& dy, float const& dz
	)
	{
		// todo resolve float to int warnings.
		length *= dx;
		width *= dy;
		height *= dz;

		xfo *= dx;
		yfo *= dy;
		zfo *= dz;
		
		//height += zfo;
		//zfo  += 2;
		
		half_length = length / 2.;
		half_width = width / 2.;
		
		sin_gamma = sin(gamma);
		cos_gamma = cos(gamma);
		
		xco = xfo + half_length*cos_gamma; // CENTER of building in QUIC domain coordinates
		yco = yfo + half_length*sin_gamma;

		this->calculateDimensionalIndices(dx, dy, dz);
		this->calculateFootprintCorners();

		this->calculatePhiPrime(ntls, dx, dy, dz);
		this->calculateUpwindDirection(ntls, dx, dy, dz);
		
		this->calculateUpwindCorners();
		this->calculateWakeCorners();
		
		this->calculateWeffLeffLr();
	}

	void urbBuilding::print() const
	{
		std::cout << "num: " 		<< bldNum 		<< ", ";
		std::cout << "group: "	<< group 	<< ", ";
		std::cout << "type: " 	<< type 	<< ", ";
		std::cout << "height: "	<< height		<< ", ";
		std::cout << "width: " 	<< width		<< ", ";
		std::cout << "length: "	<< length 	<< ", ";
		std::cout << "xfo: " 		<< xfo		<< ", ";
		std::cout << "yfo: " 		<< yfo		<< ", ";
		std::cout << "zfo: " 		<< zfo		<< ", ";
		std::cout << "gamma: "	<< gamma	<< ", ";
		std::cout << "phiprime: "			<< phiprime << ", " << std::flush;
		std::cout << "supplementalData: " 	<< supplementalData << std::endl;
	}

	bool urbBuilding::inBuildingQ(int const& i, int const& j, int const& k) const
	{
		if(i < istart || iend <= i) return false;
		if(j < jstart || jend <= j) return false;
		if(k < kstart || kend <  k) return false;
		return true;
	}

  bool urbBuilding::inBuildingQ(index3D const& loc) const
  {
    return this->inBuildingQ(loc.i, loc.j, loc.k);
  }

	void urbBuilding::interior
	(
		celltypes typs, velocities ntls, 
		float const& dx, float const& dy, float const& dz
	) const
	{
		int cell_row = typs.dim.x;
		int cell_slc = typs.dim.y*typs.dim.x;
		int grid_row = ntls.dim.x;
		int grid_slc = ntls.dim.y*ntls.dim.x;
	
		if(gamma == 0.f)
		{			
			for(int k = kstart; k <= kend; k++)
			for(int j = jstart; j <  jend; j++)
			for(int i = istart; i <  iend; i++)
			{
				int cI    = k*cell_slc + j*cell_row + i;
				int gI    = k*grid_slc + j*grid_row + i;
				int gI_pi = gI + 1;
				int gI_pj = gI + grid_row;
				int gI_pk = gI + grid_slc;
				
				typs.c[cI]    = SOLID;
				ntls.u[gI]    = ntls.v[gI]    = ntls.w[gI]    = 0.;
				ntls.u[gI_pi] = ntls.v[gI_pj] = ntls.w[gI_pk] = 0.;
			}
		}
		else
		{
			std::cerr << "Cannot set cells for rotated buildings yet." << std::endl;
		}
	}
	
	void urbBuilding::upwind
	(
		celltypes typs, velocities ntls,
		float const& dx, float const& dy, float const& dz
	)
	{		
		// Looks like upwind velocity specific stuff.
		float uhrot =  uo_h*cos_gamma + vo_h*sin_gamma;
		float vhrot = -uo_h*sin_gamma + vo_h*cos_gamma;
		
		float vel_mag = sqrt(pow(uo_h, 2.) + pow(vo_h, 2.));
		
		angle tol               = angle::E_010; // ENG
		bool perpendicular_flag = upwind_rel.isCardinalQ(tol);
		bool ns_flag            = upwind_rel.isPoleQ    (tol);
		
		float bld_prfl = (upwind_rel.isPoleQ(tol)) ? length            : width ;
		float lfcoeff  = (upwindtype == 1)         ? 2.              : 1.5 ;
		float prfl_ngl = (upwind_rel.isPoleQ(tol)) ? sin(upwind_rel) : cos(upwind_rel) ;
				
		float Lf = fabs(lfcoeff*bld_prfl*prfl_ngl / (1. + .8*bld_prfl/height));
		
		if(!this->inSubdomainQ() || !perpendicular_flag) {return;}
		
		float ynorm = fabs(yf1);
		
		static const float RETARDING_FACTOR = .4;
		static const float LENGTH_FACTOR    = .4;
		static const float HEIGHT_FACTOR    = .6;

		float retarding_height = height;
		float vortex_height    = (upwindtype == 3) ? min((float) bld_prfl, (float) height) : height ;
		
		//std::cout << "Lf = " << Lf << std::endl;
		//std::cout << "ynorm = " << ynorm << std::endl;
		
		// \\todo calculate these with relative wind direction in mind.
		// That will reduce the number of cells to concider when running through them.
		int upIstart = max(istart - rnd(1.5*Lf/dx), 1);
		int upIstop  = min(iend   + rnd(1.5*Lf/dx), (int) typs.dim.x);
		
		int upJstart = max(jstart - rnd(1.5*Lf/dy), 1);
		int upJstop  = min(jend   + rnd(1.5*Lf/dy), (int) typs.dim.y);
		
		int upKstart = max(kstart, 1);
		int upKstop  = min(rnd(HEIGHT_FACTOR*retarding_height/dz) + 1, (int) typs.dim.z);
		
		//std::cout << "upIstart = " << upIstart << " upIstop = " << upIstop << std::endl;
		//std::cout << "upJstart = " << upJstart << " upJstop = " << upIstop << std::endl;
		//std::cout << "upKstart = " << upKstart << " upKstop = " << upKstop << std::endl;

		float x_u, x_v, x_w;
		float y_u, y_v, y_w;
		
		int cell_row = typs.dim.x;
		int cell_slc = typs.dim.y*typs.dim.x;
		int grid_row = ntls.dim.x;
		int grid_slc = ntls.dim.y*ntls.dim.x;
		
	  for(int k = upKstart; k < upKstop; k++)
	  { 
			float zf = (k - .5)*dz - zfo;
			
			float sqrt_zf_vortex_height = sqrt( 1. - pow(zf/(HEIGHT_FACTOR*vortex_height),    2.) );
			float sqrt_zf_retard_height = sqrt( 1. - pow(zf/(HEIGHT_FACTOR*retarding_height), 2.) );
			
	    for(int j = upJstart; j < upJstop; j++)
			for(int i = upIstart; i < upIstop; i++)
			{
				int gI = k*grid_slc + j*grid_row + i;
				int cI = k*cell_slc + j*cell_row + i;

				bool doCell = typs.c[cI]            != 0;
				bool doU    = typs.c[cI - 1]        != 0;
				bool doV    = typs.c[cI - cell_row] != 0;
				bool doW    = typs.c[cI - cell_slc] != 0;

				determineVelocityLocations(i, j, dx, dy, upwind_dir, x_u, x_v, x_w, y_u, y_v, y_w);
	       
	      // u values
	      if(-ynorm <= y_u && y_u <= ynorm)
	      {
					float  xs_u =  ((xf2 - xf1)/(yf2 - yf1))*(y_u - yf1) + xf1;

					float Lf_factor = -Lf*sqrt( (1. - pow(y_u / ynorm, 2.)) );
					
		      float  xv_u = Lf_factor * sqrt_zf_vortex_height;
		      float xrz_u = Lf_factor * sqrt_zf_retard_height;
		      
		      float xu_sbtrct_xsu = x_u - xs_u;
		        
		      if(upwindtype == 1)
		      {
		      	if(xv_u <= xu_sbtrct_xsu && xu_sbtrct_xsu < 0.)
		      	{
           		ntls.u[gI] = 0.;
            	if(doCell) {typs.c[cI] = UPWIND;}
            }
          }
	        else if(doCell && doU)
	        {
	        	float rz_end = (zf > HEIGHT_FACTOR*vortex_height) ? 0. : LENGTH_FACTOR*xv_u ;
	        	
		      	if(xrz_u <= xu_sbtrct_xsu && xu_sbtrct_xsu < rz_end)
		      	{
		      		if(upwindtype == 3)
		      		{
	          		ntls.u[gI] *= ((xu_sbtrct_xsu - xrz_u)*(RETARDING_FACTOR - 1.)/(rz_end - xrz_u) + 1.);
	          	}
	          	else
	          	{ 
	          		ntls.u[gI] *= RETARDING_FACTOR;
	          	}
	          	typs.c[cI] = UPWIND;
		        }
		        
	         	if(LENGTH_FACTOR*xv_u <= xu_sbtrct_xsu && xu_sbtrct_xsu < 0.)
	         	{
	         		float urot =  ntls.u[gI]*cos_gamma;
	            float vrot = -ntls.u[gI]*sin_gamma;
	            
	   					// \\todo get a better name for this.
	   					float upwind_velocity_factor 
	   						=  (-HEIGHT_FACTOR*cos(((M_PI*zf)/(.5*vortex_height))) + .05)
	                *(-HEIGHT_FACTOR*sin(((M_PI*fabs(xu_sbtrct_xsu))/(LENGTH_FACTOR*Lf)) + 0.));
	   					
	            if(ns_flag)
	            {
	            	vrot = -vhrot	* upwind_velocity_factor;
	            }
	            else
	            {
	            	urot = -uhrot	* upwind_velocity_factor;
	            }
	            ntls.u[gI] = urot*cos(-gamma) + vrot*sin(-gamma);
	            typs.c [cI] = UPWIND;
	          }
        		checkVelocityMagnitude("U", "upwind", ntls.u[gI], gI);
	        }
        }
				
				
				// v values
				if(-ynorm <= y_v && y_v <= ynorm)
				{
		      float  xs_v =  ((xf2 - xf1)/(yf2 - yf1))*(y_v - yf1) + xf1;

					float Lf_factor = -Lf*sqrt( (1. - pow(y_v/ynorm, 2.)) );

          float  xv_v = Lf_factor * sqrt_zf_vortex_height;
          float xrz_v = Lf_factor * sqrt_zf_retard_height;
          
          float xv_sbtrct_xsv = x_v - xs_v;
          
          float rz_end = (zf >= HEIGHT_FACTOR*vortex_height) ? 0. : LENGTH_FACTOR*xv_v ;
          
          if(upwindtype == 1)
         	{
          	if(xv_v <= xv_sbtrct_xsv && xv_sbtrct_xsv < 0.)
						{
							ntls.v[gI]   = 0.;
							if(doCell) {typs.c[cI] = UPWIND;}
						}
					}
          else if(doCell && doV)
          {
						if(xrz_v <= xv_sbtrct_xsv && xv_sbtrct_xsv < rz_end)
						{
							if(upwindtype == 3)
							{
								ntls.v[gI] *= ((xv_sbtrct_xsv - xrz_v)*(RETARDING_FACTOR - 1.)/(rz_end - xrz_v) + 1.);
							}
							else
							{
								ntls.v[gI] *= RETARDING_FACTOR;
							}
							typs.c[cI] = UPWIND;
           	}
           	if(LENGTH_FACTOR*xv_v <= xv_sbtrct_xsv && xv_sbtrct_xsv < 0.)
           	{
           		float urot = ntls.v[gI]*sin_gamma;
              float vrot = ntls.v[gI]*cos_gamma;
              
              // \\todo find a better name.
              float upwind_velocity_factor 
              	=  (-HEIGHT_FACTOR*cos(((M_PI*zf)/(.5*vortex_height))) + .05)
                  *(-HEIGHT_FACTOR*sin(((M_PI*fabs(xv_sbtrct_xsv))/(LENGTH_FACTOR*Lf)) + 0.));
              
              if(ns_flag)
              {
              	vrot = -vhrot	* upwind_velocity_factor;
							}
							else
							{
								urot = -uhrot	* upwind_velocity_factor;
							}
							
							ntls.v[gI] = -urot*sin(-gamma) + vrot*cos(-gamma);
							typs.c [cI] = UPWIND;
						}
						checkVelocityMagnitude("V", "upwind", ntls.v[gI], gI);
					}
				}
				
				
				// w values				
				if(-ynorm <= y_w && y_w < ynorm)
				{
					float xs_w =  ((xf2 - xf1)/(yf2 - yf1))*(y_w - yf1) + xf1;
					float xv_w = -Lf*sqrt( (1. - pow(y_w/ynorm, 2.)) )*sqrt_zf_vortex_height;
					
					float xw_sbtrct_xsw = x_w - xs_w;
					
					if(upwindtype == 1)
					{
						if(xv_w <= xw_sbtrct_xsw && xw_sbtrct_xsw < 0.)
						{
							ntls.w[gI] = 0.;
							if(doCell) {typs.c[cI] = UPWIND;}
						}
					}
					else if(doCell && doW)
					{
						if(xv_w <= xw_sbtrct_xsw && xw_sbtrct_xsw < LENGTH_FACTOR*xv_w)
						{
							ntls.w[gI] *= RETARDING_FACTOR;
							typs.c[cI]  = UPWIND;
						}

						if(LENGTH_FACTOR*xv_w <= xw_sbtrct_xsw && xw_sbtrct_xsw < 0.)
						{
							ntls.w[gI] = -vel_mag*(.1*cos(((M_PI*fabs(xw_sbtrct_xsw))/(LENGTH_FACTOR*Lf)))-.05);
							typs.c[cI] =  UPWIND;
						}
						checkVelocityMagnitude("W", "upwind", ntls.v[gI], gI);
					}
				}
			}
		}
	}

	void urbBuilding::rooftop
	(
		celltypes typs, velocities ntls,
		float const& dx, float const& dy, float const& dz
	)
	{		
		angle tol               = angle::E_030;
		bool perpendicular_flag = upwind_rel.isCardinalQ(tol);
		bool ns_flag            = upwind_rel.isPoleQ    (tol);
		
		// floating point versus double percision may present descrepencies when
		// compared to fortran

		float xfront = (upwind_rel.inDirQ( WESTNORTHWEST, angle::E_090)) ? half_length : -half_length ;
		float yfront = (upwind_rel.inDirQ(SOUTHSOUTHWEST, angle::E_090)) ? half_width : -half_width ;
		
		angle xnorm = (upwind_rel.inDirQ(NORTHEAST, angle::E_015)) ? angle(gamma) + angle::E_180 :
									(upwind_rel.inDirQ(SOUTHEAST, angle::E_015)) ? angle(gamma) - angle::E_180 :
									(upwind_rel.inDirQ(SOUTHWEST, angle::E_015)) ? angle(gamma) : 
									(upwind_rel.inDirQ(NORTHWEST, angle::E_015)) ? angle(gamma) :
																													       angle::E_000 ;
																																		
		angle ynorm = (upwind_rel.inDirQ(NORTHEAST, angle::E_015)) ? angle(gamma) - angle::E_090 :
									(upwind_rel.inDirQ(SOUTHEAST, angle::E_015)) ? angle(gamma) + angle::E_090 :
									(upwind_rel.inDirQ(SOUTHWEST, angle::E_015)) ? angle(gamma) + angle::E_090 : 
									(upwind_rel.inDirQ(NORTHWEST, angle::E_015)) ? angle(gamma) - angle::E_090 :
																												                 angle::E_000 ;

		angle roofangle = (upwind_rel.inDirQ(NORTHEAST, angle::E_015)) ? calculateRoofAngle( angle::E_000,  angle::E_045) :
											(upwind_rel.inDirQ(SOUTHEAST, angle::E_015)) ? calculateRoofAngle( angle::E_000, -angle::E_045) :
											(upwind_rel.inDirQ(SOUTHWEST, angle::E_015)) ? calculateRoofAngle(-angle::E_090, -angle::E_135) : 
											(upwind_rel.inDirQ(NORTHWEST, angle::E_015)) ? calculateRoofAngle( angle::E_090,  angle::E_135) :
																															angle::E_000 ;

		//std::cout << "xfront = " << xfront << std::endl;
		//std::cout << "yfront = " << yfront << std::endl;
		//std::cout << "xnorm  = " << xnorm  << std::endl;
		//std::cout << "ynorm  = " << ynorm  << std::endl;
		//std::cout << "roofang= " << roofangle << std::endl;

		int k_ref = kend + int(.5*height/dz);		
		bool in_subdomain = (k_ref > (int) typs.dim.z) ? false : this->inSubdomainQ() && upwindtype > 0 ;
	
		if(in_subdomain)
		{
			float Bs = min(Weff, (float) height);
			float BL = max(Weff, (float) height);
			
			float Rscale = pow(Bs, 2./3.)*pow(BL, 1./3.);
			
			float Rcx = .9*Rscale;
			float vd  = .5*.22*Rscale;
			
			int kendv = kend + 1 + rnd(vd/dz); // Changed Wilson Prameter for Hc

			int rooftype_temp = (rooftype == 2 && doRooftop) ? 2 :
													(rooftype == 0)              ? 0 :
																												 1 ;
			
			//std::cout << "xfront     = " << xfront << std::endl;
      //std::cout << "yfront     = " << yfront << std::endl;
      //std::cout << "xnorm      = " << xnorm << " = " << xnorm.radians() << std::endl;
      //std::cout << "ynorm      = " << ynorm << " = " << ynorm.radians() << std::endl;
      //std::cout << "roofangle  = " << roofangle << " = " << roofangle.radians() << std::endl;
      //std::cout << "kend       = " << kend << std::endl;
      //std::cout << "k_ref      = " << k_ref << std::endl;
      //std::cout << "kendv      = " << kendv << std::endl;
      //std::cout << "rooftype      = " << rooftype << std::endl;
      //std::cout << "doRooftop     = " << doRooftop << std::endl;
      //std::cout << "roofflag_temp = " << rooftype_temp << std::endl;
      //std::cout << "ns_flag       = " << ns_flag << std::endl;
      //std::cout << "perpendicular_flag = " << perpendicular_flag << std::endl;
			
			float x_u = 0.; float x_v = 0.;	float x_w = 0.;
			float y_u = 0.;	float y_v = 0.; float y_w = 0.;
			
			float hd    = 0.;
	    float zref2 = 0.;
					
			int k_shell     = 0;
			int k_shell_ndx = 0;
			
			// Copy the initial velocities, including ALL the data we don't need.
			// \\ todo find a bettter way...
			int ttl_lmnts  = ntls.dim.x*ntls.dim.y*ntls.dim.z;
			float* uo_rgnl = new float[ttl_lmnts];
			float* vo_rgnl = new float[ttl_lmnts];
			for(int c = 0; c < ttl_lmnts; c++)
			{
				uo_rgnl[c] = ntls.u[c];
				vo_rgnl[c] = ntls.v[c];
			}
			
			int cell_row = typs.dim.x;
			int cell_slc = typs.dim.y*typs.dim.x;
			int grid_row = ntls.dim.x;
			int grid_slc = ntls.dim.y*ntls.dim.x;
			
			// Start applying the rooftop recirculation velocity modifications.
			for(int k = kend + 1; k <= k_ref; k++)
			{
				float zr = (k - kend - .5)*dz;
				
				int roofndx = k*grid_slc + rnd(yco/dy)*grid_row + rnd(xco/dx);
				
				float vel_mag = 
					sqrt
					( 
						pow(uo_rgnl[roofndx], 2.) +
				    pow(vo_rgnl[roofndx], 2.)
				  );
				  
				for(int j = jstart; j < jend; j++)
				for(int i = istart; i < iend; i++)
				{
					int gI = k*grid_slc + j*grid_row + i;
					int cI = k*cell_slc + j*cell_row + i;

			    if(typs.c[cI] == SOLID) {continue;}

			    // Check to see if velocity vector is above the building or in a street canyon cell					
 					int k_end_ndx = kend*cell_slc + j*cell_row + i;
 					
					bool NotCanyonCell = typs.c[cI]   != CANYON;
					bool k_end_isBuild = typs.c[k_end_ndx] == SOLID;
					
			  	bool uflag = NotCanyonCell && (k_end_isBuild || typs.c[k_end_ndx - 1]        == SOLID);
			  	bool vflag = NotCanyonCell && (k_end_isBuild || typs.c[k_end_ndx - cell_row] == SOLID);
			  	bool wflag = NotCanyonCell &&  k_end_isBuild;

 					determineVelocityLocations(i, j, dx, dy, angle(gamma), x_u, x_v, x_w, y_u, y_v, y_w);
			    
			    bool doU = typs.c[cI - 1]        != SOLID;
			    bool doV = typs.c[cI - cell_row] != SOLID;
			    bool doW = typs.c[cI - cell_slc] != SOLID;
			    
			    // \\todo Can cases be pushed together?
			    switch(rooftype_temp)
			    {
			    	case 1:
			    		if(uflag && doU)
			    		{
			    			hd = (!perpendicular_flag) ? min(fabs(x_u - xfront), fabs(y_u - yfront)) :
			    					 						 (ns_flag) ? fabs(y_u - yfront) : 
			    					 						 						 fabs(x_u - xfront) ;
			    						          
								zref2 = vd*sqrt(hd) / sqrt(.5*Rcx);

								k_shell     = rnd(zref2/dz) + kend;
								k_shell_ndx = k_shell*grid_slc + j*grid_row + i;			
								
								if(zr <= zref2)
								{
									ntls.u[gI] = ntls.u[k_shell_ndx]*log(zr/zo) / log(dz*(k_shell - kend - .5)/zo);
									if(wflag) {typs.c[cI] = ROOFTOP;}
									
									checkVelocityMagnitude("U", "rooftop", ntls.u[gI], gI);
								}
							}
							if(vflag && doV)
							{
								hd = (!perpendicular_flag) ? min(fabs(x_v - xfront), fabs(y_v - yfront)) :
			    					 						 (ns_flag) ? fabs(y_v - yfront) : 
			    					 						 						 fabs(x_v - xfront) ;
			    			
							
								zref2 = vd*sqrt(hd) / sqrt(.5*Rcx);
							
								k_shell     = rnd(zref2/dz) + kend;
								k_shell_ndx = k_shell*grid_slc + j*grid_row + i;
							
								if(zr <= zref2)
								{								
									ntls.v[gI] = ntls.v[k_shell_ndx]*log(zr/zo) / log(dz*(k_shell - kend - .5)/zo);
									if(wflag) {typs.c[cI] = ROOFTOP;}
									
									checkVelocityMagnitude("V", "rooftop", ntls.v[gI], gI);
								}
							}
						break;
						
		        case 2:
		        	if(perpendicular_flag)
		        	{
		        		if(k <= kendv)
		        		{
		        			if(uflag && doU)
		        			{
		        				hd = (ns_flag) ? fabs(y_u - yfront) : fabs(x_u - xfront) ;
		        			
				      			zref2 = vd*sqrt(hd) / sqrt(.5*Rcx);
				      			
				      			k_shell     = rnd(zref2/dz) + kend;
				      			k_shell_ndx = k_shell*grid_slc + j*grid_row + i;
				      			
				      			float shell_height = vd*sqrt( 1. - pow((.5*Rcx - hd)/(.5*Rcx), 2.) );
				      			
					    			if(zr <= zref2)
					    			{
					    				ntls.u[gI] = ntls.u[k_shell_ndx]*log(zr/zo) / log(dz*(k_shell - kend - .5)/zo);
											if(wflag) {typs.c[cI] = ROOFTOP;}
										}
										if(hd < Rcx && zr <= shell_height)
										{
											ntls.u[gI] = -uo_rgnl[gI]*fabs((shell_height - zr)/vd);
											if(wflag) {typs.c[cI] = ROOFTOP;}
										}
										checkVelocityMagnitude("U", "rooftop", ntls.u[gI], gI);
									}
								
									if(vflag && doV)
									{
										hd = (ns_flag) ? fabs(y_v - yfront) : fabs(x_v - xfront) ;									
								
										zref2 = vd*sqrt(hd) / sqrt(.5*Rcx);
								
										k_shell     = rnd(zref2/dz) + kend;
										k_shell_ndx = k_shell*grid_slc + j*grid_row + i;
								
										float shell_height = vd*sqrt(1. - pow( (.5*Rcx - hd)/(.5*Rcx), 2. ));

										if(zr <= zref2)
										{
											ntls.v[gI] = ntls.v[k_shell_ndx]*log(zr/zo) / log(dz*(k_shell - kend - .5)/zo);
											if(wflag) {typs.c[cI] = ROOFTOP;}
										}
									
										if(hd < Rcx && zr <= shell_height)
										{
											ntls.v[gI] = -vo_rgnl[gI]*fabs((shell_height - zr)/vd);
			            		if(wflag) {typs.c[cI] = ROOFTOP;}
										}
										checkVelocityMagnitude("V", "rooftop", ntls.v[gI], gI);											
									}
								
									if(wflag)
									{
										hd = (ns_flag) ? fabs(y_w - yfront) :	fabs(x_w - xfront) ;
								
										float shell_height = vd*sqrt( 1. - pow((.5*Rcx - hd)/(.5*Rcx), 2.) );
										
										if(hd < Rcx && zr <= shell_height)
										{
											typs.c[cI] = ROOFTOP;
										}
									}
								}
							}
							else // perp_flag
							{
								float hx = fabs(x_u - xfront);
	              float hy = fabs(y_u - yfront);
	              
	              if(hx <= min(Rcx, 2.f*hy*tan(roofangle)))
	              {
	              	if(zr <= min(Rcx, hy*tan(roofangle)))
	              	{
	              		if(uflag && doU) {ntls.u[gI] = vel_mag*cos(xnorm);}
	              		if(vflag && doV) {ntls.v[gI] = vel_mag*sin(xnorm);}
	                }
	                else if(zr <= min(Rcx, 2.f*hy*tan(roofangle)))
	                {
	                	if(uflag && doU) {ntls.u[gI] = vel_mag*cos(xnorm + angle(M_PI));}
	                	if(vflag && doV) {ntls.v[gI] = vel_mag*sin(xnorm + angle(M_PI));}
	                }
								}

	              if(hy <= min(Rcx, 2.f*hx*tan(roofangle)))
	              {
	              	if(zr <= min(Rcx, hx*tan(roofangle)))
	              	{
	              		if(uflag && doU) {ntls.u[gI] = vel_mag*cos(ynorm);}
	            			if(vflag && doV) {ntls.v[gI] = vel_mag*sin(ynorm);}
	              	}
	              	else if(zr <= min(Rcx, 2.f*hx*tan(roofangle)))
	              	{
	              		if(uflag && doU) {ntls.u[gI] = vel_mag*cos(ynorm + angle(M_PI));}
	                	if(vflag && doV) {ntls.v[gI] = vel_mag*sin(ynorm + angle(M_PI));}
	              	}
	              }
	              checkVelocityMagnitude("U", "rooftop", ntls.u[gI], gI);
            		checkVelocityMagnitude("V", "rooftop", ntls.v[gI], gI);
		            
		            if(wflag && doW)
		            {
		            	hx = fabs(x_w - xfront);
		            	hy = fabs(y_w - yfront);
		         // \\ todo remove the !inBuildingQs, apparently this is a bug.
		         // The vortices should be in the first slice above a building.
		         // 6/12/2009 ADL - according to ERP.
		              hd = hy*tan(roofangle);		                           
		              if(hx <= min(Rcx, 2.f*hd) && zr <= min(Rcx, 2.f*hd))// && !inBuildingQ(i,j,k-1))
		              {
		              	ntls.w[gI] = .1*vel_mag*((hd - hx)/hd)*(1. - fabs((zr - hd)/hd));
	              		typs.c [cI] = ROOFTOP;
		              }
		              
		              hd = hx*tan(roofangle);		              
		              if(hy <= min(Rcx, 2.f*hd) && zr <= min(Rcx, 2.f*hd))// && !inBuildingQ(i,j,k-1))
		              {
		              	ntls.w[gI] = .1*vel_mag*((hd - hy)/hd)*(1. - fabs((zr - hd)/hd));
                		typs.c [cI] = ROOFTOP;
		              }
		              checkVelocityMagnitude("W", "rooftop", ntls.w[gI], gI);
		            }
		          }
		        break;
		        default:;
		      }
			  }
			}
			
			delete [] uo_rgnl;
			delete [] vo_rgnl;
		}
	}

	void urbBuilding::wake
	(
		celltypes typs, velocities ntls,
		float const& dx, float const& dy, float const& dz
	) // const for effective dims ???
	{
		float eff_height = height;
		float dxy = min(dx,dy);
		
		bool perpendicular_flag = upwind_rel.isCardinalQ(angle(.01, DEG)) ;

		if(rooftype == 2)
		{
			bool roof_perpendicular = upwind_rel.isCardinalQ(angle::E_030);
			bool ns_flag            = upwind_rel.isPoleQ    (angle::E_030);
			
			doRooftop = true;
			
			this->searchUpwindForRooftopDisruptor(typs, dx, dy);
			
			// Potentially modify the Effective height.
			if(doRooftop && roof_perpendicular)
			{				 
				float hd = (ns_flag) ? width : length ;

				float Bs = min(Weff, (float) height);
				float BL = max(Weff, (float) height);

				float Rscale = pow(Bs, 2./3.)*pow(BL, 1./3.);
				float Rcx    = .9*Rscale;
				float vd     = .5*.22*Rscale;
			
				if(hd < Rcx)
				{
					float shell_height = vd*sqrt( 1. - pow((.5*Rcx - hd) / (.5*Rcx), 2.) );
					eff_height += (shell_height > 0) ? shell_height : 0. ;
					this->calculateLr(eff_height);
				}
			}
		}


		// Potentially modify the cav_fac and wake_fac
		// depending on the wake type. For MVP wakes.
		float cav_fac  = (waketype == 2) ? 1.1 : 1. ;
		float wake_fac = (waketype == 2) ?  .1 : 0. ;
		
		int cell_row = typs.dim.x;
		int cell_slc = typs.dim.y*typs.dim.x;
		int grid_row = ntls.dim.x;
		int grid_slc = ntls.dim.y*ntls.dim.x;

		bool* u_farwake_done = new bool[grid_slc];
		bool* v_farwake_done = new bool[grid_slc];
		
		if(this->inSubdomainQ())
		{			
			int kk = rnd(.75*kend);

			for(int k = zfo + rnd(eff_height/dz); k >= kstart; k--)
			{
				float zb = (k - .5)*dz - zfo;

				for(int c = 0; c < grid_slc; c++)
				{
					u_farwake_done[c] = v_farwake_done[c] = false;
				}	
				 
				for(int y_idx = 1; y_idx <= 2*int((yw1 - yw3)/dxy) - 1; y_idx++)
				{
					float yc = .5*y_idx*dxy + yw3;
					
				  float xwall = (perpendicular_flag) ?  xw1 : 
				  							(yc >= yw2)          ? (xw2 - xw1)/(yw2 - yw1)*(yc - yw1) + xw1 : 
				  																		 (xw3 - xw2)/(yw3 - yw2)*(yc - yw2) + xw2 ;
				  
					float ynorm         = (yc >= 0.) ? yw1 : yw3 ;
			    float canyon_factor = 1.;
						
					// Check for building that will disrupt the wake.			    
			    for(int x_idx = 1; x_idx <= int(Lr/dxy) + 1; x_idx++)
			    {
			    	float xc = x_idx*dxy;
			      
			      int i = int(((xc + xwall)*cos_upwind_dir - yc*sin_upwind_dir + xco) / dx);
			      int j = int(((xc + xwall)*sin_upwind_dir + yc*cos_upwind_dir + yco) / dy);

			      if(i >= (int) typs.dim.x - 1 || i <= 0 || j >= (int) typs.dim.y - 1 || j <= 0) {break;}
			 
			 			int kk_cI = kk*cell_slc + j*cell_row + i;
			 
			      if(!inBuildingQ(i, j, kk) && typs.c[kk_cI] == SOLID)
			      {
			      	canyon_factor = xc/Lr;
			      	break;
			      }
			    }
			    
			    
			    // Calculate and label near and far wake.
			    float dN = sqrt( (1. - pow(yc/ynorm, 2.))*(1. - pow(zb/eff_height, 2.))*pow(canyon_factor*Lr, 2.) );
			    int x_idx_min = -1;
			    
					for(int x_idx = 0; x_idx <= 2*int(FARWAKE_FAC*dN/dxy) + 1; x_idx++)
					{
			    	float xc = .5*x_idx*dxy;
			    	
			    	float i_sub = ((xc + xwall)*cos_upwind_dir - yc*sin_upwind_dir + xco) / dx;
			    	float j_sub = ((xc + xwall)*sin_upwind_dir + yc*cos_upwind_dir + yco) / dy;
			    	
			    	int i = int(i_sub);
			    	int j = int(j_sub);
			    	
			    	int iu = rnd(i_sub);
						int ju = j;
						
			    	int iv = i;
						int jv = rnd(j_sub);
						
						int udndx = ju*grid_row + iu;
						int vdndx = jv*grid_row + ju;

						int gI   = k*grid_slc +  j*grid_row + i;
						int gI_u = k*grid_slc + ju*grid_row + iu;
						int gI_v = k*grid_slc + jv*grid_row + iv;
						
						int cI = k*cell_slc + j*cell_row + i;
						
						if(i >= (int) typs.dim.x - 1 || i <= 0 || j >= (int) typs.dim.y - 1 || j <= 0) {break;}
						
						if(typs.c[cI] != SOLID && x_idx_min < 0)
						{
							x_idx_min = x_idx;
						}
						
						if(typs.c[cI] == SOLID)
						{
							if(x_idx_min >= 0)
							{
								if(inBuildingQ(i, j, k)) // Check if this building.
								{
									x_idx_min = -1;
								}
								else if(canyon_factor < 1.) {break;}
							}
						}
						
						if(typs.c[cI] == SOLID) {continue;}
						
						bool doU = typs.c[cI - 1]        != SOLID;
						bool doV = typs.c[cI - cell_row] != SOLID;
						
					// u and v values
					// Far wake
						if(xc > dN)
						{
							float farcomponent = (1. - pow(dN/(xc + wake_fac*dN), FARWAKE_EXP));
							float farwake_u = ntls.u[gI_u]*farcomponent;
							float farwake_v = ntls.v[gI_v]*farcomponent;
						
							if
							(
								typs.c[cI]    != NEARWAKE && 
								canyon_factor == 1.
							)
							{
								if
								(
									doU && !u_farwake_done[udndx] && 
									fabs(farwake_u) < fabs(ntls.u[gI_u])
								) 
								{
									ntls.u[gI_u] = farwake_u;
									ntls.w[gI]   = 0.;	
									typs.c[cI]   = FARWAKE;
									
									u_farwake_done[udndx] = true;
								}
								
								if
								(
									doV && !v_farwake_done[vdndx] && 
									fabs(farwake_v) < fabs(ntls.v[gI_v])
								)
								{
									ntls.v[gI_v] = farwake_v;
									ntls.w[gI]   = 0.;
									typs.c[cI]   = FARWAKE;
									
									v_farwake_done[vdndx] = true;
								}									
							}
						}
					// Cavity or near wake
						else
						{
							//float near_component = min(pow(1. - xc/(cav_fac*dN), 2.), 1.);
							float M1 = min(pow(1. - xc/(cav_fac*dN), 2.), 1.);
							float M2 = min(sqrt(1. - fabs(yc/ynorm)),1.);
							float near_component = M1*M2;
							
							if(doU) {ntls.u[gI_u] = -uo_h * near_component;}
							if(doV) {ntls.v[gI_v] = -vo_h * near_component;}
							
							if(doU || doV)
							{
								ntls.w[gI] = 0.;
								typs.c[cI] = NEARWAKE;
							}
						}								
						checkVelocityMagnitude("U", "wake", ntls.u[gI], gI);
						checkVelocityMagnitude("V", "wake", ntls.u[gI], gI);
					}
				}
			}
		}
		
		delete [] u_farwake_done;
		delete [] v_farwake_done;
	}

	void urbBuilding::canyon
	(
		celltypes typs, velocities const& ntls, std::vector<urbBuilding> const& bldngs,
		float const& dx, float const& dy,	float const& dz
	) const
	{
		float dxy = min(dx,dy);

		angle angle_tol = angle( 135., DEG); // ENG RAD
		angle crdnl_tol = angle(  .01, DEG); // ENG
		angle ordnl_tol = angle(44.99, DEG); // ENG
		
		bool perpendicular_flag = upwind_rel.isCardinalQ(crdnl_tol);
		
		// Calculate upwind direction specific to the canyon
		angle upwind_norm  = (upwind_rel.inDirQ(NORTH, crdnl_tol)) ? angle(gamma) + angle::E_090 :
												 (upwind_rel.inDirQ(EAST,  crdnl_tol)) ? angle(gamma)         :
												 (upwind_rel.inDirQ(SOUTH, crdnl_tol)) ? angle(gamma) - angle::E_090 :
												 																				 angle(gamma) - angle::E_180 ;
		
		angle upwind_norm1 = (upwind_rel.inDirQ(NORTHEAST, ordnl_tol)) ? angle(gamma) + angle::E_090 :
									 			 (upwind_rel.inDirQ(SOUTHEAST, ordnl_tol)) ? angle(gamma)         :
									 			 (upwind_rel.inDirQ(SOUTHWEST, ordnl_tol)) ? angle(gamma) - angle::E_090 :
									 			 (upwind_rel.inDirQ(NORTHWEST, ordnl_tol)) ? angle(gamma) + angle::E_180 :
																																  			     angle::E_000 ;
         
    angle upwind_norm2 = (upwind_rel.inDirQ(NORTHEAST, ordnl_tol)) ? angle(gamma)         :
									 			 (upwind_rel.inDirQ(SOUTHEAST, ordnl_tol)) ? angle(gamma) - angle::E_090 :
									 			 (upwind_rel.inDirQ(SOUTHWEST, ordnl_tol)) ? angle(gamma) - angle::E_180 :
									 			 (upwind_rel.inDirQ(NORTHWEST, ordnl_tol)) ? angle(gamma) + angle::E_090 :
																																    			   angle::E_000 ;
		int cell_row = typs.dim.x;
		int cell_slc = typs.dim.y*typs.dim.x;
		int grid_row = ntls.dim.x;
		int grid_slc = ntls.dim.y*ntls.dim.x;

		angle along_dir  = angle::E_000;
		angle cross_dir  = angle::E_000;
		angle canyon_dir = angle::E_000;
		float velmag     = 0.;
		int   k_ref      = 0;

		for(int y_idx = 1; y_idx <= 2*int((yw1 - yw3)/dxy) - 1; y_idx++)
		{
    	float yc       = .5*y_idx*dxy + yw3;
      bool  top_flag = false;
      
      for(int k = kend; k >= kstart; k--)
      {
      	float zc = (k - .5)*dz;
         
        float xwall = (perpendicular_flag) ?  xw1 : 
				 							(yc >= yw2)          ? (xw2 - xw1)/(yw2 - yw1)*(yc - yw1) + xw1 : 
				  																	 (xw3 - xw2)/(yw3 - yw2)*(yc - yw2) + xw2 ;
				  																	 
        upwind_norm = (perpendicular_flag) ? upwind_norm  :
        							(yc >= yw2) 				 ? upwind_norm1 : 
        																		 upwind_norm2 ; 

        bool canyon_flag  = false;
        bool reverse_flag = false;
        
        float S = 0.;
        
        int x_idx_min = -1;  
        int x_idx_max =  0;
         
        for(int x_idx = 0; x_idx <= 2*int(Lr/dxy) + 1; x_idx++)
        {
         	float xc = .5*x_idx*dxy;
         	          
          int i = rnd(((xc + xwall)*cos_upwind_dir - yc*sin_upwind_dir + xco) / dx - .5);
          int j = rnd(((xc + xwall)*sin_upwind_dir + yc*cos_upwind_dir + yco) / dy - .5);
          
          int gI = k*grid_slc + j*grid_row + i;          
          int cI = k*cell_slc + j*cell_row + i;

          if(i >= (int) ntls.dim.x - 1 || i <= 0 || j >= (int) ntls.dim.y - 1 || j <= 0) {break;}
          
          if(typs.c[cI] != SOLID && x_idx_min < 0)
          {
          	x_idx_min = x_idx;
          }

          if(typs.c[cI] == SOLID && x_idx_min >= 0)
          {
          	canyon_flag = true;
          	x_idx_max   = x_idx - 1;
						
						// Length from parent building to any wake building along x-wake dimension.
						S = .5*(x_idx_max - x_idx_min)*dxy;
												
						if(!top_flag)
						{						
							k_ref = k + 1;

							// Check these
							int ic = rnd(((.5*x_idx_max*dxy + xwall)*cos_upwind_dir - yc*sin_upwind_dir + xco) / dx);
							int jc = rnd(((.5*x_idx_max*dxy + xwall)*sin_upwind_dir + yc*cos_upwind_dir + yco) / dy);

							int vndx_c_kref = k_ref*grid_slc + jc*grid_row + ic;							
							int cndx_c_kref = k_ref*cell_slc + jc*cell_row + ic;
							
							if(typs.c[cndx_c_kref] != SOLID)
							{				
								if
								(
									typs.c[cndx_c_kref - 1]        != SOLID && 
									typs.c[cndx_c_kref - cell_row] != SOLID
								)
								{
									velmag     = sqrt( pow(ntls.u[vndx_c_kref], 2.) + pow(ntls.v[vndx_c_kref], 2.) );
									canyon_dir = angle(atan2(ntls.v[vndx_c_kref], ntls.u[vndx_c_kref])); // ENG RAD
								}
								else if(typs.c[cndx_c_kref - 1] != SOLID)
								{
									velmag     = fabs(ntls.u[vndx_c_kref]);
									canyon_dir = (ntls.u[vndx_c_kref] > 0.) ? angle::E_000 :  angle::E_180 ;
								}	
								else
								{
									velmag     = fabs(ntls.v[vndx_c_kref]);
									canyon_dir = (ntls.v[vndx_c_kref] > 0.) ? angle::E_090 : -angle::E_090 ;
								}
								top_flag = true;
							}
							else
							{
								canyon_flag = top_flag = false;
								S = 0.;
								break;
							}
							
							if(checkVelocityMagnitude("velmag", "canyon", velmag, gI))
							{									
								canyon_flag = top_flag = false;
								S = 0.;
								break;
							}						

						// Find the along canyon and cross canyon directions
							urbBuilding* dwnwnd_bldng = findBuilding(bldngs, i, j, k);
						
							if(dwnwnd_bldng->height < height)
							{
								canyon_flag = top_flag = false;
								S = 0.;
								break;
							}
						
							if(dwnwnd_bldng != NULL)
							{
								// \\todo why the slight change?
								i = rnd(((xc-.5*dxy + xwall)*cos_upwind_dir - yc*sin_upwind_dir + xco) / dx + .5);
			          j = rnd(((xc-.5*dxy + xwall)*sin_upwind_dir + yc*cos_upwind_dir + yco) / dy + .5);						

								dwnwnd_bldng->determineCanyonDirections
								(
									i, j, dx, dy, canyon_dir, 
									along_dir, cross_dir, reverse_flag
								);
							}
						}
						
						if(reverse_flag)
						{
							if(cos(cross_dir - upwind_norm) < -cos(angle_tol))
							{
								canyon_flag = top_flag = false;
								S = 0.;
							}
						}
						else
						{
							if(cos(cross_dir - upwind_norm) > cos(angle_tol))
							{
								canyon_flag = top_flag = false;
								S = 0.;
							}
						}

						break;
          }
        }
        
				if(canyon_flag && S > 0.9*dxy)
				{
					float cross_mag = fabs(velmag*cos(canyon_dir - cross_dir));
					float along_mag = fabs(velmag*cos(canyon_dir - along_dir))*log(zc/zo)/log(dz*(k_ref - .5)/zo);

					checkVelocityMagnitude("along", "canyon", along_mag, k*grid_slc); // was gI
					checkVelocityMagnitude("cross", "canyon", cross_mag, k*grid_slc); // was gI

					float sin_along_dir = sin(along_dir);					
					float cos_along_dir = cos(along_dir);
					
					float sin_cross_dir = sin(cross_dir);
					float cos_cross_dir = cos(cross_dir);
					
					for(int x_idx = x_idx_min; x_idx <= x_idx_max; x_idx++)
					{
						float xc = .5*x_idx*dxy;
						
						int i = rnd(((xc + xwall)*cos_upwind_dir - yc*sin_upwind_dir + xco)/dx - .5);
						int j = rnd(((xc + xwall)*sin_upwind_dir + yc*cos_upwind_dir + yco)/dy - .5);
						
						int gI = k*grid_slc + j*grid_row + i;
						int cI = k*cell_slc + j*cell_row + i;
						
						float xpos    = fabs(.5*(x_idx - x_idx_min))*dxy;
						float xpOverS = xpos / S;
						
						float cross_mag_factor = cross_mag*xpOverS*4.*(1. - xpOverS);
						
						bool doU = typs.c[cI - 1]          != SOLID;
						bool doV = typs.c[cI - cell_row]   != SOLID;
						bool doW = typs.c[cI - cell_slc] != SOLID;
						
						if(typs.c[cI] != SOLID)
						{
							// u component						
							if(doU) 
							{
								ntls.u[gI] = along_mag*cos_along_dir + cross_mag_factor*cos_cross_dir;
							}

							// v component
							if(doV) 
							{
								ntls.v[gI] = along_mag*sin_along_dir + cross_mag_factor*sin_cross_dir;
							}
						
							// w component
							if(doW)
							{
								ntls.w[gI]  = fabs(.5*cross_mag*(1. - 2.*xpOverS))*(2.*xpOverS - 1.);
								ntls.w[gI] *= (reverse_flag) ? 1. : -1. ;
							}
							typs.c[cI] = CANYON;

							checkVelocityMagnitude("U", "canyon", ntls.u[gI], gI);
							checkVelocityMagnitude("V", "canyon", ntls.v[gI], gI);
							checkVelocityMagnitude("W", "canyon", ntls.w[gI], gI);
						}
					}
				}
      }
   	}
   	//*/
	}

	void urbBuilding::calculatePhiPrime(velocities const& ntls, float const& dx, float const& dy, float const& dz)
	{
		// \\todo uo_h and vo_h are the same as cu and cv, right?
		// PUT THEM TOGETHER, only one of them is needed!!.
		// Shouldn't these use xco and yco?
		int xndx = rnd((xfo + length/2.) / dx) + 1; // \\todo check these. Should they all lose 1.?
		int yndx = rnd(yfo  / dy) + 1;
		int zndx = rnd(height / dz) + 2;
	
		int   ndx = zndx*ntls.dim.y*ntls.dim.x + yndx*ntls.dim.x + xndx;
		float cu  = ntls.u[ndx];
		float cv  = ntls.v[ndx];
		
		angle theta = angle::M_000; // MET
		
		// Is there a better way to do this?
		if(cu == 0 && cv >  0) {theta = angle::M_180;}
		if(cu == 0 && cv <= 0) {theta = angle::M_000;}
		
		// Temp angles are in engineering, but everything works out.
		if(cu <  0 && cv <  0) {theta = angle::M_090 - angle(atan( cv/cu));}
		if(cu <  0 && cv >= 0) {theta = angle::M_090 + angle(atan(-cv/cu));}
		if(cu >  0 && cv >= 0) {theta = angle::M_270 - angle(atan( cv/cu));}
		if(cu >  0 && cv <  0) {theta = angle::M_270 + angle(atan(-cv/cu));}
		
		// Is there a better way to do this?		
		angle phi = (theta > angle::M_270) ?  theta - angle::M_270 : 
		                                     -theta + angle::M_270 ;
		
		phiprime  =	(phi < angle::M_090) ? ( phi - angle::M_000) :
								(phi < angle::M_180) ? (-phi + angle::M_180) :
								(phi < angle::M_270) ? ( phi - angle::M_180) :
																       (-phi + angle::M_360) ;
	}

	void urbBuilding::calculateDimensionalIndices(float const& dx, float const& dy, float const& dz)
	{
		// \\todo Do each of these need 1 taken off? Indexing starts at 0 now...
		
		// Domain reference indices of start and end
		// of building in each dimension.
		istart = rnd(xfo / dx);
		iend   = istart + int(length / dx);
				
		jstart = rnd((yfo - half_width) / dy);
		jend   = jstart + int(width / dy);
				
		kstart = rnd(zfo / dz) + 1;
		kend   = kstart + int(height / dz) - 1; // Actually one less than terminating index.
		
		// \\todo check for buildings outside of domain to properly set indices.
		
		// debugging ingo
		//std::cout << "istart = " << istart << " iend = " << iend << std::endl;
		//std::cout << "jstart = " << jstart << " jend = " << jend << std::endl;
		//std::cout << "kstart = " << kstart << " kend = " << kend << std::endl;
	}

	void urbBuilding::calculateFootprintCorners()
	{
		// Location of corners relative to the center of the building.
	
		// Which corner is this?
		x1 = xfo + half_width*sin_gamma - xco;
		y1 = yfo - half_width*cos_gamma - yco;
		
		// Which corner is this?
		x2 = x1 + length*cos_gamma;
		y2 = y1 + length*sin_gamma;
		
		// Which corner is this?
		x4 = xfo - half_width*sin_gamma - xco;
		y4 = yfo + half_width*cos_gamma - yco;		
		
		// Which corner is this? // Leave last.
		x3 = x4 + length*cos_gamma;
		y3 = y4 + length*sin_gamma;
	}

	void urbBuilding::calculateUpwindDirection
	(
		velocities const& ntls, 
		float const& dx, 
		float const& dy, 
		float const& dz
	)
	{	
		// find upwind direction and determine the type of flow regime.

		// \\todo uo_h and vo_h are the same as cu and cv, right?
		// PUT THEM TOGETHER, only one of them is needed!!.
		// Grab the velocity centered directly above the building
		int gI = (kend + 1)*ntls.dim.y*ntls.dim.x + rnd(yco/dy)*ntls.dim.x + rnd(xco/dx);
		uo_h   = ntls.u[gI];
		vo_h   = ntls.v[gI];

		// Find the upwind and calculate its sin and cos to save time.		
		upwind_dir     = angle(atan2(vo_h, uo_h));
		sin_upwind_dir = sin(upwind_dir);
		cos_upwind_dir = cos(upwind_dir);

		// Find the upwind_rel angle for this building.
		upwind_rel = upwind_dir - angle(gamma);
		
		// debugging info
		//std::cout << "uo_h = " << uo_h << std::endl;
		//std::cout << "vo_h = " << vo_h << std::endl;
		//std::cout << "upwind_dir = " << upwind_dir << " = " << upwind_dir.radians() << std::endl;
		//std::cout << "upwind_rel = " << upwind_rel << " = " << upwind_rel.radians() << std::endl;
	}

	void urbBuilding::calculateUpwindCorners()
	{
		angle tol = angle::E_010;
	
		if(upwind_rel.inDirQ(NORTH, tol))
		{
			xf1 =  x2*cos_upwind_dir + y2*sin_upwind_dir;
			yf1 = -x2*sin_upwind_dir + y2*cos_upwind_dir;
			xf2 =  x1*cos_upwind_dir + y1*sin_upwind_dir;
			yf2 = -x1*sin_upwind_dir + y1*cos_upwind_dir;
		}
		else if(upwind_rel.inDirQ(EAST, tol))
		{
			xf1 =  x1*cos_upwind_dir + y1*sin_upwind_dir;
			yf1 = -x1*sin_upwind_dir + y1*cos_upwind_dir;
			xf2 =  x4*cos_upwind_dir + y4*sin_upwind_dir;
			yf2 = -x4*sin_upwind_dir + y4*cos_upwind_dir;
		}
		else if(upwind_rel.inDirQ(SOUTH, tol))
		{
			xf1 =  x4*cos_upwind_dir + y4*sin_upwind_dir;
			yf1 = -x4*sin_upwind_dir + y4*cos_upwind_dir;
			xf2 =  x3*cos_upwind_dir + y3*sin_upwind_dir;
			yf2 = -x3*sin_upwind_dir + y3*cos_upwind_dir;
		}
		else if(upwind_rel.inDirQ(WEST, tol))
		{
			xf1 =  x3*cos_upwind_dir + y3*sin_upwind_dir;
			yf1 = -x3*sin_upwind_dir + y3*cos_upwind_dir;
			xf2 =  x2*cos_upwind_dir + y2*sin_upwind_dir;
			yf2 = -x2*sin_upwind_dir + y2*cos_upwind_dir;
		}
	}
	
	void urbBuilding::calculateWakeCorners()
	{
		angle crdnl_tol = angle(  .01, DEG);
		angle ordnl_tol = angle(44.99, DEG);
		
		if(upwind_rel.inDirQ(NORTH, crdnl_tol))
		{
			xw1 =  x4*cos_upwind_dir + y4*sin_upwind_dir;
			yw1 = -x4*sin_upwind_dir + y4*cos_upwind_dir;
			xw3 =  x3*cos_upwind_dir + y3*sin_upwind_dir;
			yw3 = -x3*sin_upwind_dir + y3*cos_upwind_dir;
		}
		else if(upwind_rel.inDirQ(NORTHEAST, ordnl_tol))
		{
			xw1 =  x4*cos_upwind_dir + y4*sin_upwind_dir;
			yw1 = -x4*sin_upwind_dir + y4*cos_upwind_dir;
			xw2 =  x3*cos_upwind_dir + y3*sin_upwind_dir;
			yw2 = -x3*sin_upwind_dir + y3*cos_upwind_dir;
			xw3 =  x2*cos_upwind_dir + y2*sin_upwind_dir;
			yw3 = -x2*sin_upwind_dir + y2*cos_upwind_dir;
		}
		else if(upwind_rel.inDirQ(EAST, crdnl_tol))
		{
			xw1 =  x3*cos_upwind_dir + y3*sin_upwind_dir;
			yw1 = -x3*sin_upwind_dir + y3*cos_upwind_dir;
			xw3 =  x2*cos_upwind_dir + y2*sin_upwind_dir;
			yw3 = -x2*sin_upwind_dir + y2*cos_upwind_dir;
		}
		else if(upwind_rel.inDirQ(SOUTHEAST, ordnl_tol))
		{
			xw1 =  x3*cos_upwind_dir + y3*sin_upwind_dir;
			yw1 = -x3*sin_upwind_dir + y3*cos_upwind_dir;
			xw2 =  x2*cos_upwind_dir + y2*sin_upwind_dir;
			yw2 = -x2*sin_upwind_dir + y2*cos_upwind_dir;
			xw3 =  x1*cos_upwind_dir + y1*sin_upwind_dir;
			yw3 = -x1*sin_upwind_dir + y1*cos_upwind_dir;
		}
		else if(upwind_rel.inDirQ(SOUTH, crdnl_tol))
		{
			xw1 =  x2*cos_upwind_dir + y2*sin_upwind_dir;
			yw1 = -x2*sin_upwind_dir + y2*cos_upwind_dir;
			xw3 =  x1*cos_upwind_dir + y1*sin_upwind_dir;
			yw3 = -x1*sin_upwind_dir + y1*cos_upwind_dir;
		}
		else if(upwind_rel.inDirQ(SOUTHWEST, ordnl_tol))
		{
			xw1 =  x2*cos_upwind_dir + y2*sin_upwind_dir;
			yw1 = -x2*sin_upwind_dir + y2*cos_upwind_dir;
			xw2 =  x1*cos_upwind_dir + y1*sin_upwind_dir;
			yw2 = -x1*sin_upwind_dir + y1*cos_upwind_dir;
			xw3 =  x4*cos_upwind_dir + y4*sin_upwind_dir;
			yw3 = -x4*sin_upwind_dir + y4*cos_upwind_dir;
		}
		else if(upwind_rel.inDirQ(WEST, crdnl_tol))
		{
			xw1 =  x1*cos_upwind_dir + y1*sin_upwind_dir;
			yw1 = -x1*sin_upwind_dir + y1*cos_upwind_dir;
			xw3 =  x4*cos_upwind_dir + y4*sin_upwind_dir;
			yw3 = -x4*sin_upwind_dir + y4*cos_upwind_dir;
		}
		else if(upwind_rel.inDirQ(NORTHWEST, ordnl_tol))
		{
			xw1 =  x1*cos_upwind_dir + y1*sin_upwind_dir;
			yw1 = -x1*sin_upwind_dir + y1*cos_upwind_dir;
			xw2 =  x4*cos_upwind_dir + y4*sin_upwind_dir;
			yw2 = -x4*sin_upwind_dir + y4*cos_upwind_dir;
			xw3 =  x3*cos_upwind_dir + y3*sin_upwind_dir;
			yw3 = -x3*sin_upwind_dir + y3*cos_upwind_dir;
		}		
		else
		{
			std::cerr << "building.wake() - You shouldn't get here." << std::endl;
			std::cerr << "If so, then something may be wrong in angle class." << std::endl;
		}
		
		//std::cout << "wc1 = (" << xw1 << ", " << yw1 << ")" << std::endl;
		//std::cout << "wc2 = (" << xw2 << ", " << yw2 << ")" << std::endl;
		//std::cout << "wc3 = (" << xw3 << ", " << yw3 << ")" << std::endl;
	}

	void urbBuilding::calculateWeffLeffLr()
	{
		// Potentially modify the Effective length based on wind angle.

		angle upwind_rel_norm = upwind_rel + angle::E_090;
		angle beta            = angle(fabs(atan2(length, width)));
		
		if
		(
			upwind_rel.isPoleQ(beta)
	
			//fabs(upwind_rel.radians()) > (angle(.5*M_PI) - beta).radians() && 
			//fabs(upwind_rel.radians()) < (angle(.5*M_PI) + beta).radians()
		)
		{
			Leff = fabs(width/sin(upwind_rel));
		}
		else
		{
			Leff = fabs(length/cos(upwind_rel));
		}
		
		if(waketype == 2)
		{
			if
			(
				upwind_rel_norm.isPoleQ(beta)
			
				//fabs(upwind_rel_norm.radians()) > (angle(.5*M_PI) - beta).radians() && 
				//fabs(upwind_rel_norm.radians()) < (angle(.5*M_PI) + beta).radians()
			)
			{
				Weff = fabs(width/sin(upwind_rel_norm));
			}
			else
			{
				Weff = fabs(length/cos(upwind_rel_norm));
			}
		}
		else
		{
			float fabs_phiprime_gamma = fabs(phiprime.radians() - gamma);
			Weff = length*sin(fabs_phiprime_gamma) + width*cos(fabs_phiprime_gamma);
		}
		this->calculateLr(height);
	}

	void urbBuilding::calculateLr(float const& _height)
	{
		// Length over Height and Width over Height ratios.
		float LoverH = Leff / _height;
		float WoverH = Weff / _height;
		
		if(LoverH > 3.)  {LoverH = 3.;}
		if(LoverH < .3)  {LoverH = .3;}
		if(WoverH > 10.) {WoverH = 10.;}
		
		Lr = (1.8*_height*WoverH) / (pow(LoverH, .3)*(1. + .24*WoverH));		
	}

	bool urbBuilding::inSubdomainQ() const
	{
		bool n_sbdmn = false;
		
		// Generically test to see if building is in subdomain.
		if
		(
			true
			//x_subdomain_start <= xfo - dx && xfo - dx <  x_subdomain_end &&
			//y_subdomain_start <= yfo - dy && yfo - dy <= y_subdomain_end
		) 
		{
			n_sbdmn = true;
		}
		
		return n_sbdmn;
	}

	void urbBuilding::determineVelocityLocations
	(
		int const& i, int const& j, 
		float const& dx, float const& dy,
		angle const& angle,
		float& x_u, float& x_v, float& x_w,
		float& y_u, float& y_v, float& y_w
	) const
	{
    x_u =  ((i - 0.)*dx - xco)*cos(angle) + ((j + .5)*dy - yco)*sin(angle);
    y_u = -((i - 0.)*dx - xco)*sin(angle) + ((j + .5)*dy - yco)*cos(angle);
    
    x_v =  ((i + .5)*dx - xco)*cos(angle) + ((j - 0.)*dy - yco)*sin(angle);
    y_v = -((i + .5)*dx - xco)*sin(angle) + ((j - 0.)*dy - yco)*cos(angle);
   
   	x_w =  ((i + .5)*dx - xco)*cos(angle) + ((j + .5)*dy - yco)*sin(angle);
    y_w = -((i + .5)*dx - xco)*sin(angle) +	((j + .5)*dy - yco)*cos(angle);
  }

	void urbBuilding::determineCanyonDirections
	(
		int const& i, int const& j, float const& dx, float const& dy,
		angle const& canyon_dir, angle& along_dir, angle& cross_dir, 
		bool& reverse_flag
	) const
	{
		float xcd = xfo + half_length*cos_gamma;
		float ycd = yfo + half_length*sin_gamma;
		
		float xd =  ((i - .5)*dx - xcd)*cos_gamma + ((j - .5)*dy - ycd)*sin_gamma;
		float yd = -((i - .5)*dx - xcd)*sin_gamma + ((j - .5)*dy - ycd)*cos_gamma;

		angle beta         = angle(fabs(atan2(length, width))); // ENG RAD
		angle thetad       = angle(atan2(yd,xd));
		angle downwind_rel = canyon_dir - angle(gamma);
														
		// \\todo the following section should be rewritten and should look much
		// better if done with the angle class methods.
		/*
		angle NS_tol = angle(fabs(atan2(length,width)));
		angle EW_tol = angle::E_090 - NS_tol;
		
		if(thetad.inDirectionQ(NORTH, NS_tol)
		{
			
		}
		else if(thetad.inDirectionQ(EAST, EW_tol)
		{
		
		}
		else if(thetad.inDirectionQ(SOUTH, NS_tol)
		{
		
		}
		else if(thetad.inDirectionQ(WEST, EW_tol)
		{
		
		}
		else
		{
			std::cerr << "Error in building::determineCanyonDirections()." << std::endl;
			std::cerr << "thetad not being categorized as N, S, E or W." << std::endl;
		}
		*/
		if(angle::E_090 - beta <= thetad && thetad <= angle::E_090 + beta)
		{
			if(downwind_rel <= angle::E_000)
			{
				if(downwind_rel <= -angle::E_090)
				{
					along_dir = -angle::E_180 + angle(gamma);
					cross_dir = along_dir - angle::E_090;
				}
				else
				{
					along_dir = angle(gamma);
					cross_dir = along_dir + angle::E_090;
				}
			}
			else
			{
				reverse_flag = true;
				
				if(downwind_rel >= angle::E_090)
				{
					along_dir = -angle::E_180 + angle(gamma);
					cross_dir = along_dir + angle::E_090;
				}
				else
				{
					along_dir = angle(gamma);
					cross_dir = along_dir - angle::E_090;
				}
			}
		}
	 	else if(-angle::E_090 + beta < thetad && thetad < angle::E_090 - beta)
	 	{
			if(-downwind_rel <= -angle::E_090 || angle::E_090 <= downwind_rel) //fabs(downwind_rel) >= angle::E_090)
			{
				if(downwind_rel < angle::E_000)
				{
					along_dir = -angle::E_090 + angle(gamma);
					cross_dir = along_dir + angle::E_090;
				}
				else
				{
					along_dir = angle::E_090 + angle(gamma);
					cross_dir = along_dir - angle::E_090;
				}
			}
			else
			{
				reverse_flag = false;
				
				if(downwind_rel < angle::E_000)
				{
					along_dir = -angle::E_090 + angle(gamma);
					cross_dir = along_dir - angle::E_090;
				}
				else
				{
					along_dir = angle::E_090 + angle(gamma);
					cross_dir=along_dir + angle::E_090;
				}
			}
		}
		else if(thetad <= -angle::E_090 + beta && thetad >= -angle::E_090 - beta)
		{
			if(downwind_rel >= angle::E_000)
			{
				if(downwind_rel <= angle::E_090)
				{
					along_dir = angle(gamma);
					cross_dir = along_dir - angle::E_090;
				}
				else
				{
					along_dir = -angle::E_180 + angle(gamma);
					cross_dir = along_dir + angle::E_090;
				}
			}
			else
			{
				reverse_flag = true;
				
				if(downwind_rel >= angle::E_090)
				{
					along_dir = angle(gamma);
					cross_dir = along_dir + angle::E_090;
				}
				else
				{
					along_dir = -angle::E_180 + angle(gamma);
					cross_dir = along_dir - angle::E_090;
				}
			}
		}
		else
		{
			if(-angle::E_090 < downwind_rel || downwind_rel < angle::E_090) //fabs(downwind_rel) < angle::E_090)
			{
				if(downwind_rel >= angle::E_000)
				{
					along_dir = angle::E_090 + angle(gamma);
					cross_dir = along_dir + angle::E_090;
				}
				else
				{
					along_dir = -angle::E_090 + angle(gamma);
					cross_dir = along_dir - angle::E_090;
				}
			}
			else
			{
				reverse_flag = true;
				
				if(downwind_rel >= angle::E_000)
				{
					along_dir = angle::E_090 + angle(gamma);
					cross_dir = along_dir - angle::E_090;
				}
				else
				{
					along_dir = -angle::E_090 + angle(gamma);
					cross_dir = along_dir + angle::E_090;
				}
			}
		}
 	}

	void urbBuilding::searchUpwindForRooftopDisruptor
	(
		celltypes typs, 
		float const& dx, float const& dy
	)
	{
		doRooftop        = true;
    int   upwind_cnt = 0;
    float dxy        = min(dx, dy);
    
    for(int y_idx = 1; y_idx < 2*int((yw1 - yw3)/dxy) - 1; y_idx++)
    {
			float yc = .5*y_idx*dxy + yw3;
			
			bool perpendicular_flag = upwind_rel.isCardinalQ(angle(.01, DEG));
			
			float xwall = (perpendicular_flag) ?  xf2 :
										(yc >= yf2)					 ? (xf2 - xw1)/(yf2 - yw1)*(yc - yw1) + xw1 :
																					 (xw3 - xf2)/(yw3 - yf2)*(yc - yf2) + xf2 ;

			for(int x_idx = int(Lr/dxy) + 1; x_idx >= 1; x_idx--)
			{
				float xc = -x_idx*dxy;
				
				int i = int(((xc + xwall)*cos_upwind_dir - yc*sin_upwind_dir + xco) / dx);
				int j = int(((xc + xwall)*sin_upwind_dir + yc*cos_upwind_dir + yco) / dy);
				
				int cI = kend*typs.dim.x*typs.dim.y + j*typs.dim.x + i;
				
				if(1 <= i && i <= (int) typs.dim.x && 1 <= j && j <= (int) typs.dim.y)
				{
					if(typs.c[cI] == SOLID && !inBuildingQ(i,j,kend)) {upwind_cnt++; break;}
				}
			}
    }
    if(upwind_cnt >= int((yw1 - yw3)/dxy)) {doRooftop = false;}
    //std::cout << "upwind_cnt = " << upwind_cnt << std::endl;
    //std::cout << "int((yw1 - yw3)/dxy) = " << int((yw1 - yw3)/dxy) << std::endl;
    //std::cout << "doRooftop = " << doRooftop << std::endl;
	}

	angle urbBuilding::calculateRoofAngle(angle const& a1, angle const& a2) const
	{		
		return 
			angle
			(
				.0513f
				*
				exp
				(
					1.7017f
					*
					(
						fabs(a1.radians() - upwind_rel.radians()) 
						- 
						2.f*fabs(a2.radians() - upwind_rel.radians())
					)
				)
			);
	}

	void swapBuildings(urbBuilding* b1, urbBuilding* b2)
	{
		urbBuilding* tmp = b1;
		b1 = b2;
		b2 = tmp;
		// Swapped.
	}
	
	void sortBuildings(std::vector<urbBuilding*> bldngs)
	{
		int veggies = 0;
		for(unsigned int j = 0; j < bldngs.size(); j++)
		{
			for(unsigned int i = 0; i < bldngs.size() - j; i++)
			{
				if(bldngs[i]->type == quBuildings::VEGETATION)
				{
					swapBuildings(bldngs[i], bldngs[j]);
					veggies++;				
				}
				else if(bldngs[i]->height > bldngs[j]->height)
				{
					swapBuildings(bldngs[i], bldngs[j]);
				}
			}
		}
		// Vegetation at the back.		
	}
	
	urbBuilding* findBuilding
	(
		std::vector<urbBuilding> bldngs, 
		int const& i, int const& j, int const& k
	)
	{
		for(unsigned int n = 0; n < bldngs.size(); n++)
		{
			if(bldngs[n].inBuildingQ(i, j, k))
			{
				return &bldngs[n];
			}
		}
		return NULL;
	}
}

