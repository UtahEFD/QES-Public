#pragma once

/*
 * This class is a container relating to sensors and metric
 * information read from the xml.
 */
#include <algorithm>

#include "util/ParseInterface.h"
#include "Sensor.h"

class MetParams : public ParseInterface
{
private:



public:

	bool metInputFlag;
	int num_sites;
	int maxSizeDataPoints;
	int z0_domain_flag;
	std::string siteName;
	std::string fileName;
	std::vector<Sensor*> sensors;



	virtual void parseValues()
	{
		parsePrimitive<bool>(true, metInputFlag, "metInputFlag");
		parsePrimitive<int>(true, num_sites, "num_sites");
		parsePrimitive<int>(true, maxSizeDataPoints, "maxSizeDataPoints");
		parsePrimitive<int>(true, z0_domain_flag, "z0_domain_flag");
		parsePrimitive<std::string>(false, siteName, "siteName");
		parsePrimitive<std::string>(false, fileName, "fileName");
		parseMultiElements<Sensor>(false, sensors, "sensor");

	}
};
