#pragma once

#include "util/ParseInterface.h"
#include "Sensor.h"

class MetParams : public ParseInterface
{
private:



public:

	bool metInputFlag;
	int num_sites;
	int maxSizeDataPoints;
	std::string siteName;
	std::string fileName;
	std::vector<Sensor*> sensors;
	int site_coord_flag, site_UTM_zone;
	float site_UTM_x, site_UTM_y;
	float site_lon, site_lat;


	virtual void parseValues()
	{
		parsePrimitive<bool>(true, metInputFlag, "metInputFlag");
		parsePrimitive<int>(true, num_sites, "num_sites");
		parsePrimitive<int>(true, maxSizeDataPoints, "maxSizeDataPoints");
		parsePrimitive<std::string>(true, siteName, "siteName");
		parsePrimitive<std::string>(true, fileName, "fileName");
		parseMultiElements<Sensor>(true, sensors, "sensor");

  	parsePrimitive<int>(true, site_coord_flag, "site_coord_flag");
		parsePrimitive<float>(false, site_UTM_x, "site_UTM_x");
		parsePrimitive<float>(false, site_UTM_y, "site_UTM_y");
		parsePrimitive<int>(false, site_UTM_zone, "site_UTM_zone");
		parsePrimitive<float>(false, site_lon, "site_lon");
		parsePrimitive<float>(false, site_lat, "site_lat");

	}
};
