#pragma once

#include "ParseInterface.h"

class MetParams : public ParseInterface
{
private:
	bool metInputFlag;
	int numMeasuringSites;
	int maxSizeDataPoints;
	std::string siteName;
	std::string fileName;


public:

	virtual void parseValues()
	{
		parsePrimative<int>(true, verticalStretching, "verticalStretching");
		parsePrimative<bool>(true, numMeasuringSites, "numMeasuringSites");
		parsePrimative<int>(true, maxSizeDataPoints, "maxSizeDataPoints");
		parsePrimative<std::string>(true, siteName, "siteName");
		parsePrimative<std::string>(true, fileName, "fileName");

	}
};