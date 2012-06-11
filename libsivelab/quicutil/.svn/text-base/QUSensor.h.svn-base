#ifndef __QUICDATAFILE_QUSENSOR_H__
#define __QUICDATAFILE_QUSENSOR_H__ 1

#include "QUICDataFile.h"

#include "../util/angle.h"

// //////////////////////////////////////////////////////////////////
// 
// Class for holding the QU_metparams.inp file
// 
// //////////////////////////////////////////////////////////////////
class quSensorParams : public quicDataFile
{
public:

  enum ProfileType {
    LOG = 1,
    EXP = 2,
    CANOPY = 3,
    DATAPT = 4   
  };
//  enum BOUNDARY {LOG = 1, EXP = 2, URBAN_CANOPY = 3, DISCRETE = 4};

	int time_steps;
		
	int prfl_lgth;
		
	float* u_prof;
	float* v_prof;
		
  std::string siteName;
  std::string fileName;

  int xCoord;
  int yCoord;
  
  float decimalTime;
  ProfileType boundaryLayerFlag;

  float exponential; // new exp, exp / zo NOTE: Check this.
  float Zo; // new zo, site_pp,  NOTE: Check this.
  float recipMoninObukhovLen; // new rL
  float canopyHeight; // new H
  float attenuationCoef; // new ac

/* These look like equivilants from above. If you're missing something, look here.  
  float zo; // ? site_pp exp or zo OR exp / zo
	float pp; // aka site zo building roughness??	// extend for more time steps
	float rL;																			// extend for more time steps
	float H;																			// extend for more time steps
	float ac;																			// extend for more time steps
*/

  float siteExponential;

  float height, speed;
  sivelab::angle direction;

public:  
  quSensorParams()
  : quicDataFile()
  {}
  
  ~quSensorParams() {}

public:
  bool readQUICFile(const std::string &filename);
  bool writeQUICFile(const std::string &filename);
  
	void print() const;
	void determineVerticalProfiles(float const& zo, float const& dz, int const& i_time = 0);


private:

	static float calculatePSI_M(float const& value, bool const& basic);
};

#endif // #define __QUICDATA_QUSENSOR_H__ 1
