#pragma once

#include "ParseInterface.h"
#include "Sensor.h"
<<<<<<< HEAD

=======
/*
 *Placeholder class for parsed met parameters info in the xml
 */
>>>>>>> origin/doxygenAdd
class MetParams : public ParseInterface
{
private:



public:

	bool metInputFlag;
<<<<<<< HEAD
	int num_sites;
	int maxSizeDataPoints;
	std::string siteName;
	std::string fileName;
	std::vector<Sensor*> sensors;
=======
	int numMeasuringSites;
	int maxSizeDataPoints;
	std::string siteName;
	std::string fileName;
	Sensor* sensor;
>>>>>>> origin/doxygenAdd


	virtual void parseValues()
	{
		parsePrimitive<bool>(true, metInputFlag, "metInputFlag");
<<<<<<< HEAD
		parsePrimitive<int>(true, num_sites, "num_sites");
		parsePrimitive<int>(true, maxSizeDataPoints, "maxSizeDataPoints");
		parsePrimitive<std::string>(true, siteName, "siteName");
		parsePrimitive<std::string>(true, fileName, "fileName");
		parseMultiElements<Sensor>(true, sensors, "sensor");

	}
};
=======
		parsePrimitive<int>(true, numMeasuringSites, "numMeasuringSites");
		parsePrimitive<int>(true, maxSizeDataPoints, "maxSizeDataPoints");
		parsePrimitive<std::string>(true, siteName, "siteName");
		parsePrimitive<std::string>(true, fileName, "fileName");
		parseElement<Sensor>(true, sensor, "sensor");

	}
};
>>>>>>> origin/doxygenAdd
