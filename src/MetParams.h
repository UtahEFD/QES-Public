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

	int z0_domain_flag = 0;
	std::vector<Sensor*> sensors;



	virtual void parseValues()
	{
		parsePrimitive<int>(false, z0_domain_flag, "z0_domain_flag");
		parseMultiElements<Sensor>(false, sensors, "sensor");
	}
};
