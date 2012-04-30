#include "sensor.h"

namespace QUIC
{
	sensor::sensor() : wnd_drctn(NORTH, MET)
	{
		name = file = "";
		
		x = y = 0;
		
		time = 0.;
		
		boundary = LOG;
		
		pp = rL = H = ac = 0.;

		time_steps = 0;
		
		wnd_hght = wnd_spd = 0;
		
		prfl_lgth = 0;
		
		u_prof = v_prof = NULL;
	}
	
	sensor::~sensor()
	{
		delete [] u_prof;
		delete [] v_prof;
	}


	void sensor::print() const
	{
		std::cout << "name = " << name << ", ";
		std::cout << "file = " << file << ", ";
		
		std::cout << "x = " << x << ", ";
		std::cout << "y = " << y << ", ";
		
		std::cout << "time = " << time << ", ";
		
		std::cout << "boundary = " << boundary << ", ";
		
		std::cout << "pp = " << pp << ", ";
		std::cout << "rL = " << rL << ", ";
		
		std::cout << "H = " << H << ", ";			// extend for more time steps
		std::cout << "ac = " << ac << ", ";		// extend for more time steps

		std::cout << "time_steps = " << time_steps << ", ";
		
		std::cout << "height = " << wnd_hght << ", ";
		std::cout << "speed = " << wnd_spd << ", ";
		std::cout << "direction = " << wnd_drctn << std::endl;

	}
	
	void sensor::determineVerticalProfiles(float const& zo, float const& dz, int const& i_time)
	{
		if(i_time != 0)
		{
			// \\todo get this code working for more than on time value.
			// the time steps portion was left out.
			std::cerr << "A time other than zero was specficied for determining wind profiles for a sensor." << std::endl;
		}
	
		if(boundary == DISCRETE)
		{
			 /*allocate(u_data(site_nz_data(kk,i_time)),v_data(site_nz_data(kk,i_time)))				 
			 do ii=1,site_nz_data(kk,i_time)
			    // ! convert wind speed wind direction into u and v
			    if(site_wd_data(kk,i_time,ii)<=90) then
			       u_data(ii)=-site_ws_data(kk,i_time,ii)*sin((site_wd_data(kk,i_time,ii))*M_PI/180.)
			       v_data(ii)=-site_ws_data(kk,i_time,ii)*cos((site_wd_data(kk,i_time,ii))*M_PI/180.)
			    endif
			    if(site_wd_data(kk,i_time,ii)>90&&site_wd_data(kk,i_time,ii)<=180) then
			       u_data(ii)=-site_ws_data(kk,i_time,ii)*cos((site_wd_data(kk,i_time,ii)-90)*M_PI/180.)
			       v_data(ii)= site_ws_data(kk,i_time,ii)*sin((site_wd_data(kk,i_time,ii)-90)*M_PI/180.)
			    endif
			    if(site_wd_data(kk,i_time,ii)>180&&site_wd_data(kk,i_time,ii)<=270) then
			       u_data(ii)= site_ws_data(kk,i_time,ii)*sin((site_wd_data(kk,i_time,ii)-180)*M_PI/180.)
			       v_data(ii)= site_ws_data(kk,i_time,ii)*cos((site_wd_data(kk,i_time,ii)-180)*M_PI/180.)
			    endif
			    if(site_wd_data(kk,i_time,ii)>270&&site_wd_data(kk,i_time,ii)<=360) then
			       u_data(ii)= site_ws_data(kk,i_time,ii)*cos((site_wd_data(kk,i_time,ii)-270)*M_PI/180.)
			       v_data(ii)=-site_ws_data(kk,i_time,ii)*sin((site_wd_data(kk,i_time,ii)-270)*M_PI/180.)
			    endif
			 enddo
			 */
			std::cerr << "The data entry profile for sensors is not supported." << std::endl;
		}

		// Make some standard angles to be used later.
		angle a_000 = angle(  0., DEG, MET);
		angle a_090 = angle( 90., DEG, MET);
		angle a_180 = angle(180., DEG, MET);
		angle a_270 = angle(270., DEG, MET);

		// Find this sites u and v direction multipliers.
		float umult_site = (wnd_drctn <  a_000) ?  0. :
											 (wnd_drctn <= a_090) ? -sin(wnd_drctn) :
											 (wnd_drctn <= a_180) ? -cos(wnd_drctn - a_090) :
											 (wnd_drctn <= a_270) ?  sin(wnd_drctn - a_180) :
			                                         cos(wnd_drctn - a_270) ;
											 									 														 									 
		float vmult_site = (wnd_drctn <  a_000) ?  0. :
											 (wnd_drctn <= a_090) ? -cos(wnd_drctn) :
											 (wnd_drctn <= a_180) ?  sin(wnd_drctn - a_090) :
											 (wnd_drctn <= a_270) ?  cos(wnd_drctn - a_180) :
											                      	-sin(wnd_drctn - a_270) ;
											 									 		
		//std::cout << "umult_site = " << umult_site << " vmult_site = " << vmult_site << std::endl;
		

		// Set up some common variables that are used often.
    float psi_m = calculatePSI_M( wnd_hght*rL, wnd_hght*rL >= 0 );
    float ustar = wnd_spd*VON_KARMAN/(log(wnd_hght / pp) + psi_m);
    
    
    // bisect is a function described in bisect.f90.
    float d  = 0.;
    float uH = 0.;
    
    // Special case...
    if(boundary == URBAN_CANOPY)
    {
			d  = bisect(ustar, pp, H, ac, psi_m);
	    uH = (ustar / VON_KARMAN)*(log((H - d)/pp) + psi_m);
	    
   		psi_m = calculatePSI_M( (H - d)*rL, H*rL >= 0 );
    												
			if(wnd_hght > H)
			{
				psi_m = calculatePSI_M( (wnd_hght - d)*rL, wnd_hght*rL >= 0 );
			}
	
			wnd_spd /= (wnd_hght <= H) ? uH*exp(ac*(wnd_hght / H - 1.)) :
																	 (ustar / VON_KARMAN)*(log((wnd_hght - d) / zo) + psi_m) ;
	
			ustar *= wnd_spd;
			uH *= wnd_spd;
		}
	
		u_prof[0] = v_prof[0] = 0.;
		for(int k = 1; k < prfl_lgth; k++)
		{								
			float z_ij = dz*(k - .5);
		
			switch(boundary)
			{
				// power law profile
				case EXP : 
					u_prof[k] = umult_site*wnd_spd*(pow(z_ij / wnd_hght, pp));
					v_prof[k] = vmult_site*wnd_spd*(pow(z_ij / wnd_hght, pp));
				break;
		
				// logrithmic velocity profile
				case LOG :
					psi_m = calculatePSI_M( z_ij*rL, z_ij*rL >= 0 );

					u_prof[k] = (umult_site*ustar / VON_KARMAN)*(log(z_ij / pp) + psi_m);
					v_prof[k] = (vmult_site*ustar / VON_KARMAN)*(log(z_ij / pp) + psi_m);
				break;
				
				// Canopy profile
				case URBAN_CANOPY :
					if(z_ij <= H) // lower canopy profile
					{
						u_prof[k] = umult_site*uH*exp(ac*(z_ij / H - 1.));
						v_prof[k] = vmult_site*uH*exp(ac*(z_ij / H - 1.));
					}
					else // upper canopy profile
					{
						psi_m = calculatePSI_M( (z_ij - d)*rL, z_ij*rL >= 0 );
			
						u_prof[k] = (umult_site*ustar / VON_KARMAN)*(log((z_ij - d) / pp) + psi_m);
						v_prof[k] = (vmult_site*ustar / VON_KARMAN)*(log((z_ij - d) / pp) + psi_m);
					}
				break;
						
				// data entry profile
				case DISCRETE :
					std::cout << "Using DISCRETE." << std::endl;
					std::cerr << "The data entry profile for sensors is not supported." << std::endl;
			
					/*
					do ii=1,site_nz_data(kk,i_time) !loop through the data points in input profile each time step
						 z_grid=(real(k-2)+0.5)*dz	!var dz
					! begin interpolation input velocity to computational grid
						 if(z_grid==site_z_data(kk,i_time,ii))then
								u_prof(kk,k)=u_data(ii)
								v_prof(kk,k)=v_data(ii)
								goto 500
						 endif
					!erp 9/23/05	logarithmically interpolate to zero velocity at zo below lowest data point
					!MAN 01/21/07 logarithmic interpolation uses the first data point instead of the second
						 if(z_grid > 0 && z_grid < site_z_data(kk,i_time,1))then
								if(z_grid > site_pp(kk,i_time))then
									 u_prof(kk,k)= (u_data(1)/(log(site_z_data(kk,i_time,1)/site_pp(kk,i_time))))*   &
										                       log(z_grid/site_pp(kk,i_time))
									 v_prof(kk,k)= (v_data(1)/(log(site_z_data(kk,i_time,1)/site_pp(kk,i_time))))*   &
										                       log(z_grid/site_pp(kk,i_time))
								else
									 u_prof(kk,k)= 0
									 v_prof(kk,k)= 0
								endif
								goto 500
						 endif
					!erp 9/23/05
						 if(ii > 1)then
								if(z_grid > site_z_data(kk,i_time,ii-1) && z_grid<site_z_data(kk,i_time,ii))then
									 u_prof(kk,k)=((u_data(ii)-u_data(ii-1))/(site_z_data(kk,i_time,ii)-site_z_data(kk,i_time,ii-1)))   &
										               *(z_grid-site_z_data(kk,i_time,ii-1)) + u_data(ii-1)
									 v_prof(kk,k)=((v_data(ii)-v_data(ii-1))/(site_z_data(kk,i_time,ii)-site_z_data(kk,i_time,ii-1)))   &
										               *(z_grid-site_z_data(kk,i_time,ii-1)) + v_data(ii-1)
									 goto 500
								endif
						 endif
					enddo !end ii loop
					500              continue

					! extrapolate logarithmically for data beyond input velocity
					if(z_grid > site_z_data(kk,i_time,site_nz_data(kk,i_time))) then
						 u_prof(kk,k)=(log(z_grid/site_z_data(kk,i_time,site_nz_data(kk,i_time)-1))/   &
										log(site_z_data(kk,i_time,site_nz_data(kk,i_time))/site_z_data(kk,i_time,site_nz_data(kk,i_time)-1)))*   &
										(u_data(site_nz_data(kk,i_time))-   &
										u_data(site_nz_data(kk,i_time)-1)) + u_data(site_nz_data(kk,i_time)-1)
						 v_prof(kk,k)=(log(z_grid/site_z_data(kk,i_time,site_nz_data(kk,i_time)-1))/   &
										log(site_z_data(kk,i_time,site_nz_data(kk,i_time))/site_z_data(kk,i_time,site_nz_data(kk,i_time)-1)))*   &
										(v_data(site_nz_data(kk,i_time))-   &
										v_data(site_nz_data(kk,i_time)-1)) + v_data(site_nz_data(kk,i_time)-1)
					endif
					if(k == nz) deallocate(u_data,v_data)
					endif !erp 2/6/2003 end data entry
					*/
				break;
				
				default: 
					std::cerr << "An unknown boundary type (" << boundary << ") was encountered." << std::endl;
					std::cerr << "Sensor::You shouldn't be able to get here." << std::endl;
				break;
			}
		}
		
		//std::cout << "psi_m = " << psi_m << std::endl;
    //std::cout << "ustar = " << ustar << std::endl;
    //std::cout << "d     = " << d     << std::endl;
    //std::cout << "uH    = " << uH    << std::endl;
	}
	
	float sensor::calculatePSI_M(float const& value, bool const& basic)
	{
		if(basic)
		{
			return 4.7*value;
		}
		else
		{
			float xtemp = pow(1. - 15.*value, .25);
			return -2.*log(.5*(1. + xtemp)) - log(.5*(1. + pow(xtemp, 2.))) + 2.*atan(xtemp) - .5*M_PI;
		}
	}
}
