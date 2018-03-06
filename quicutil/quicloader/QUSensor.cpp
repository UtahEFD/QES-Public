#include "QUSensor.h"

#include <cstdlib>
#include <fstream>

#include "bisect.h"
#include "constants.h"
#include "legacyFileParser.h"
void quSensorParams::build_map()
{


		var_addressMap["time_steps"]=&time_steps;


	  	var_addressMap["prfl_lgth"]=&prfl_lgth;

	  	var_addressMap["siteName"]=&siteName;

		var_addressMap["fileName"]=&fileName;

	  	var_addressMap["xCoord"]=&xCoord;

	  	var_addressMap[" yCoord"]=&yCoord;

		var_addressMap["decimalTime"]=&decimalTime;

	  	var_addressMap["boundaryLayerFlag"]=&boundaryLayerFlag;

	  	var_addressMap["exponential"]=&exponential;


		var_addressMap["Zo"]=&Zo;

	  	var_addressMap["recipMoninObukhovLen"]=&recipMoninObukhovLen;

	  	var_addressMap["canopyHeight"]=&canopyHeight;

		var_addressMap["attenuationCoef"]=&attenuationCoef;


	  	var_addressMap["siteExponential"]=&siteExponential;

	  	var_addressMap[" height"]=&height;


		var_addressMap["speed"]=&speed;



                //	void* temp_ptr;
		// temp_ptr=&direction;
	//  	var_addressMap["direction"]=temp_ptr;    //uncomment this if we want anlge class to be accessed 

}
quSensorParams& quSensorParams::operator=(const quSensorParams& other)
{

  //  std::cerr<<"operator ---------quSensorParams---------"<<std::endl;
  if (this == &other)
    return *this;



  time_steps=other.time_steps;
		
  prfl_lgth=other.prfl_lgth;
		
  u_prof=other.u_prof;
  v_prof=other.v_prof;
		
  siteName=other.siteName;
  fileName=other.siteName;

  xCoord=other.xCoord;
  yCoord=other.yCoord;
  
  decimalTime=other.decimalTime;
 
  boundaryLayerFlag=other.boundaryLayerFlag; 

  exponential=other.exponential; 
  Zo=other.Zo;
  recipMoninObukhovLen=other.recipMoninObukhovLen; 
  canopyHeight=other.canopyHeight; 
  attenuationCoef=other.attenuationCoef; 

  siteExponential=other.siteExponential;

  height=other.height; speed=other.speed;

  direction=other.direction;

  return * this;
}

bool quSensorParams::readQUICFile(const std::string &filename)
{
  // Format is the same for 5.6 and 5.72 - nothing needs to change here.

  if (beVerbose)
  {
    std::cout << "\tParsing QUIC sensor file: " << filename << std::endl;
  }
  
  std::ifstream quicFile(filename.c_str(), std::ifstream::in);
  if(!quicFile.is_open())
  {
	  std::cerr << "Error in " << __FILE__ << ":" << __func__ << std::endl;
    std::cerr << "quicLoader could not open :: " << filename << "." << std::endl;
    exit(EXIT_FAILURE);
  }
		
  std::string line;
  std::stringstream ss(line, std::stringstream::in | std::stringstream::out);

  getline(quicFile, line);
  ss.str(line);
  ss >> siteName;
		
  if (quicVersionString == "5.92" 
      || quicVersionString == "6.01" 
      || quicVersionString == "6.1" ) {
      // 1 !Site Coordinate Flag (1=QUIC, 2=UTM, 3=Lat/Lon)
      getline(quicFile, line);
      ss.str(line);
      ss >> siteCoordinateFlag;
  }

  getline(quicFile, line);
  ss.str(line);
  ss >> xCoord;

  getline(quicFile, line);
  ss.str(line);
  ss >> yCoord;

  getline(quicFile, line);
  ss.str(line);
  ss >> decimalTime;

  getline(quicFile, line);
  ss.str(line);
  int profileType;
  ss >> profileType;

  //boundaryLayerFlag = (ProfileType)profileType;
  boundaryLayerFlag = profileType;
  if (boundaryLayerFlag == 1)
  {
    getline(quicFile, line);
    ss.str(line);
    ss >> Zo;
    
    getline(quicFile, line);
    ss.str(line);
    ss >> recipMoninObukhovLen;
  }
  else if (boundaryLayerFlag == 2)
  {
    getline(quicFile, line);
    ss.str(line);
    ss >> exponential;
  }
  else if (boundaryLayerFlag == 3)
  {
    getline(quicFile, line);
    ss.str(line);
    ss >> Zo;
    
    getline(quicFile, line);
    ss.str(line);
    ss >> recipMoninObukhovLen;

    getline(quicFile, line);
    ss.str(line);
    ss >> canopyHeight;
    
    getline(quicFile, line);
    ss.str(line);
    ss >> attenuationCoef;
  }
  else if (boundaryLayerFlag == 4)
  {
    std::cerr << "Do not yet support Discrete Data Points Wind Profiles yet!!!!" << std::endl;
    //exit(EXIT_FAILURE);
  }
  else
  {
    std::cerr << "Unknown Wind Profile type: " << boundaryLayerFlag << "!!! Exiting!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // step over the height, speed, direction header
  getline(quicFile, line);
  ss.str(line);

  getline(quicFile, line);
  ss.str(line);
  float directionAsFloat = 0.f;
  ss >> height >> speed >> directionAsFloat;

  //direction.setRadians(directionAsFloat, sivelab::MET); // Meteorlagical?
  // std::cout<<"the value read in "<<directionAsFloat<<"\n"; 
  direction.setDegrees(directionAsFloat, sivelab::MET);
  


  //std::cout<<"read the wind angle and this should have been 270 according to the answer "<<std::endl;
 // std::cout<<"the value is "<<direction.degrees(sivelab::MET)<<std::endl;
 // exit(1);
  quicFile.close();
  return true;
}

bool quSensorParams::writeQUICFile(const std::string &filename)
{
  std::ofstream qufile;
  qufile.open(filename.c_str());

  if (qufile.is_open())
  {
    qufile << siteName << "\t\t\t!Site Name " << std::endl;
    qufile << xCoord << "\t\t\t!X coordinate (meters)" << std::endl;
    qufile << yCoord << "\t\t\t!Y coordinate (meters)" << std::endl;
    qufile << decimalTime << "\t\t\t!Decimal time (military time i.e. 0130 = 1.5)" << std::endl;
    qufile << boundaryLayerFlag << "\t\t\t!site boundaryLayerFlag layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)" << std::endl;

    if (boundaryLayerFlag == 1)  //LOG
	  {
	    qufile << Zo << "\t\t\t!site zo" << std::endl;
	    qufile << recipMoninObukhovLen << "\t\t\t!reciprocal Monin-Obukhov Length (1/m)" << std::endl;
	  }
    else if (boundaryLayerFlag == 2)  //EXP
	  {
	    qufile << exponential << "\t\t\t!site exponential" << std::endl;
	  }
    else if (boundaryLayerFlag == 3)   //CANOPY
	  {
	    qufile << Zo << "\t\t\t!site zo" << std::endl;
	    qufile << recipMoninObukhovLen << "\t\t\t!reciprocal Monin-Obukhov Length (1/m)" << std::endl;
	    qufile << canopyHeight << "\t\t\t!Canopy height (m)" << std::endl;
	    qufile << attenuationCoef << "\t\t\t!attenuation coefficient" << std::endl;
	  }
    else if (boundaryLayerFlag == 4)   //DATAPT
	  {
	    std::cerr << "Do not yet support Discrete Data Points Wind Profiles yet!!!!" << std::endl;
	    //exit(EXIT_FAILURE);
	  }
    else
	  {
	    std::cerr << "Unknown Wind Profile type: " << boundaryLayerFlag << "!!! Exiting!" << std::endl;
	    exit(EXIT_FAILURE);
	  }

    qufile << "!Height (m),Speed	(m/s), Direction (deg relative to true N)" << std::endl;
    qufile << height << " " << speed << " " << direction.degrees(sivelab::MET)<< std::endl;

    return true;
  }

  return false;
}

void quSensorParams::determineVerticalProfiles(float const& zo, float const& dz, int const& i_time)
{
		if(i_time != 0)
		{
			// \\todo get this code working for more than on time value.
			// the time steps portion was left out.
			std::cerr << "A time other than zero was specficied for determining wind profiles for a sensor." << std::endl;
		}
	
		if(boundaryLayerFlag == 4)  //DATAPT
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
		using namespace sivelab;
		//angle a_000 = angle(  0., DEG, MET);
		//angle a_090 = angle( 90., DEG, MET);
		//angle a_180 = angle(180., DEG, MET);
		//angle a_270 = angle(270., DEG, MET);

		// Find this sites u and v direction multipliers.
		float umult_site = (direction <  angle::M_000) ?  0.f :
											 (direction <= angle::M_090) ? -sin(direction) :
											 (direction <= angle::M_180) ? -cos(direction - angle::M_090) :
											 (direction <= angle::M_270) ?  sin(direction - angle::M_180) :
			                                                cos(direction - angle::M_270) ;
											 									 														 									 
		float vmult_site = (direction <  angle::M_000) ?  0.f :
											 (direction <= angle::M_090) ? -cos(direction) :
											 (direction <= angle::M_180) ?  sin(direction - angle::M_090) :
											 (direction <= angle::M_270) ?  cos(direction - angle::M_180) :
											                      	       -sin(direction - angle::M_270) ;
											 									 		
		//std::cout << "umult_site = " << umult_site << " vmult_site = " << vmult_site << std::endl;
		

		// Set up some common variables that are used often.
    float psi_m = calculatePSI_M( height*recipMoninObukhovLen, height*recipMoninObukhovLen >= 0 );
    float ustar = speed*VON_KARMAN/(log(height / Zo) + psi_m);
    
    
    // bisect is a function described in bisect.f90.
    float d  = 0.f;
    float uH = 0.f;
    
    // Special case...
    if(boundaryLayerFlag == 3)	//CANOPY
    {
			d  = bisect(ustar, Zo, canopyHeight, attenuationCoef, psi_m);
	    uH = (ustar / VON_KARMAN)*(log((canopyHeight - d) / Zo) + psi_m);
	    
   		psi_m = calculatePSI_M( (canopyHeight - d)*recipMoninObukhovLen, canopyHeight*recipMoninObukhovLen >= 0.f );
    												
			if(height > canopyHeight)
			{
				psi_m = calculatePSI_M( (height - d)*recipMoninObukhovLen, height*recipMoninObukhovLen >= 0.f );
			}
	
			speed /= (height <= canopyHeight) ? uH*exp(attenuationCoef*(height / canopyHeight - 1.)) :
																	 (ustar / VON_KARMAN)*(log((height - d) / zo) + psi_m) ;
	
			ustar *= speed;
			uH *= speed;
		}
	
		u_prof[0] = v_prof[0] = 0.f;
		for(int k = 1; k < prfl_lgth; k++)
		{								
			float z_ij = dz*(k - .5f);
		
			switch(boundaryLayerFlag)
			{
				// power law profile
				case 2 : //EXP
					u_prof[k] = umult_site*speed*(pow(z_ij / height, Zo));
					v_prof[k] = vmult_site*speed*(pow(z_ij / height, Zo));
				break;
		
				// logrithmic velocity profile
				case 1 :  //LOG
					psi_m = calculatePSI_M( z_ij*recipMoninObukhovLen, z_ij*recipMoninObukhovLen >= 0.f );

					u_prof[k] = (umult_site*ustar / VON_KARMAN)*(log(z_ij / Zo) + psi_m);
					v_prof[k] = (vmult_site*ustar / VON_KARMAN)*(log(z_ij / Zo) + psi_m);
				break;
				
				// Canopy profile
				case 3 :  //CANOPY
					if(z_ij <= canopyHeight) // lower canopy profile
					{
						u_prof[k] = umult_site*uH*exp(attenuationCoef*(z_ij / canopyHeight - 1.f));
						v_prof[k] = vmult_site*uH*exp(attenuationCoef*(z_ij / canopyHeight - 1.f));
					}
					else // upper canopy profile
					{
						psi_m = calculatePSI_M( (z_ij - d)*recipMoninObukhovLen, z_ij*recipMoninObukhovLen >= 0.f );
			
						u_prof[k] = (umult_site*ustar / VON_KARMAN)*(log((z_ij - d) / Zo) + psi_m);
						v_prof[k] = (vmult_site*ustar / VON_KARMAN)*(log((z_ij - d) / Zo) + psi_m);
					}
				break;
						
				// data entry profile
				case 4 :	//DATAPT
					std::cout << "Using DATAPT." << std::endl;
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
								if(z_grid > site_Zo(kk,i_time))then
									 u_prof(kk,k)= (u_data(1)/(log(site_z_data(kk,i_time,1)/site_Zo(kk,i_time))))*   &
										                       log(z_grid/site_Zo(kk,i_time))
									 v_prof(kk,k)= (v_data(1)/(log(site_z_data(kk,i_time,1)/site_Zo(kk,i_time))))*   &
										                       log(z_grid/site_Zo(kk,i_time))
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
					std::cerr << "An unknown boundaryLayerFlag type (" << boundaryLayerFlag << ") was encountered." << std::endl;
					std::cerr << "Sensor::You shouldn't be able to get here." << std::endl;
				break;
			}
		}
		
		//std::cout << "psi_m = " << psi_m << std::endl;
    //std::cout << "ustar = " << ustar << std::endl;
    //std::cout << "d     = " << d     << std::endl;
    //std::cout << "uH    = " << uH    << std::endl;
	}

void quSensorParams::print() const
{
  std::cout << "name = " << siteName << ", " << std::endl;
	std::cout << "file = " << fileName << ", " << std::endl;
	
	std::cout << "x = " << xCoord << ", " << std::endl;
	std::cout << "y = " << yCoord << ", " << std::endl;
		
	std::cout << "boundaryLayerFlag = " << boundaryLayerFlag << ", " << std::endl;
	
	std::cout << "Zo = " << Zo << ", " << std::endl;
	std::cout << "recipMoninObukhovLen = " << recipMoninObukhovLen << ", " << std::endl;
	
	std::cout << "H = " << canopyHeight << ", " << std::endl;			// extend for more time steps
	std::cout << "attenuationCoef = " << attenuationCoef << ", " << std::endl;		// extend for more time steps

	std::cout << "time_steps = " << time_steps << ", " << std::endl;
	
	std::cout << "height = " << height << ", " << std::endl;
	std::cout << "speed = " << speed << ", " << std::endl;
	std::cout << "direction = " << direction.degrees(sivelab::MET) << std::endl;
}

	float quSensorParams::calculatePSI_M(float const& value, bool const& basic)
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
