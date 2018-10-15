#pragma once

#include "ParseInterface.h"
<<<<<<< HEAD
#define _USE_MATH_DEFINES
#include <math.h>


=======

/*
 *Placeholder class for parsed sensor info in the xml
 */
>>>>>>> origin/doxygenAdd
class Sensor : public ParseInterface
{
private:

public:
<<<<<<< HEAD
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


=======
		int siteCoords;
		int x, y;
		float epoch;
		bool boundaryLayerFlag;
		float siteZ0;
		int reciprocal;
		int height;
		int speed;
		int direction; 


	virtual void parseValues()
	{
		parsePrimitive<int>(true, siteCoords, "siteCoords");
		parsePrimitive<int>(true, x, "x");
		parsePrimitive<int>(true, y, "y");
		parsePrimitive<float>(true, epoch, "epoch");
		parsePrimitive<bool>(true, boundaryLayerFlag, "boundaryLayerFlag");
		parsePrimitive<float>(true, siteZ0, "siteZ0");
		parsePrimitive<int>(true, reciprocal, "reciprocal");
		parsePrimitive<int>(true, height, "height");
		parsePrimitive<int>(true, speed, "speed");
		parsePrimitive<int>(true, direction, "direction");

	}
};
>>>>>>> origin/doxygenAdd
