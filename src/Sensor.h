#pragma once

#include "ParseInterface.h"
#define _USE_MATH_DEFINES
#include <math.h>


class Sensor : public ParseInterface
{
private:

public:
		int site_blayer_flag;
		float site_one_overL;
		float site_xcoord;
		float site_ycoord;
		float site_wind_dir;

		float site_z0;
		float site_z_ref;
		float site_U_ref;
		

	virtual void parseValues()
	{
		//rsePrimitive<int>(true, num_sites, "numberofSites");
		//parsePrimitive<int>(true, siteCoords, "siteCoords");
		parsePrimitive<float>(true, site_xcoord, "site_xcoord");
		parsePrimitive<float>(true, site_ycoord, "site_ycoord");
		parsePrimitive<int>(true, site_blayer_flag, "boundaryLayerFlag");
		parsePrimitive<float>(true, site_z0, "siteZ0");
		parsePrimitive<float>(true, site_one_overL, "reciprocal");
		parsePrimitive<float>(true, site_z_ref, "height");
		parsePrimitive<float>(true, site_U_ref, "speed");
		parsePrimitive<float>(true, site_wind_dir, "direction");

	}

};


