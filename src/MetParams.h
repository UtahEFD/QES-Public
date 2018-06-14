#pragma once

#include "ParseInterface.h"
#include "Sensor.h"

class MetParams : public ParseInterface
{
private:



public:

	bool metInputFlag;
	int numMeasuringSites;
	int maxSizeDataPoints;
	std::string siteName;
	std::string fileName;
	Sensor* sensor;


	virtual void parseValues()
	{
		parsePrimative<bool>(true, metInputFlag, "metInputFlag");
		parsePrimative<int>(true, numMeasuringSites, "numMeasuringSites");
		parsePrimative<int>(true, maxSizeDataPoints, "maxSizeDataPoints");
		parsePrimative<std::string>(true, siteName, "siteName");
		parsePrimative<std::string>(true, fileName, "fileName");
		parseElement<Sensor>(true, sensor, "sensor");

	}
};