#pragma once

/*
 * This class is a container relating to sensors and metric
 * information read from the xml.
 */
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


	virtual void parseValues()
	{
		parsePrimitive<bool>(true, metInputFlag, "metInputFlag");
		parsePrimitive<int>(true, num_sites, "num_sites");
		parsePrimitive<int>(true, maxSizeDataPoints, "maxSizeDataPoints");
		parsePrimitive<std::string>(true, siteName, "siteName");
		parsePrimitive<std::string>(true, fileName, "fileName");
		parseMultiElements<Sensor>(true, sensors, "sensor");

	}
};
