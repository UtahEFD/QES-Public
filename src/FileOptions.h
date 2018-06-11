#pragma once

#include "ParseInterface.h"

class FileOptions : public ParseInterface
{
private:



public:

	int outputFormat;
	bool massConservedFlag;
	bool sensorVelocityFlag;
	bool staggerdVelocityFlag;

	virtual void parseValues()
	{
		parsePrimative<int>(true, outputFormat, "outputFormat");
		parsePrimative<bool>(true, massConservedFlag, "massConservedFlag");
		parsePrimative<bool>(true, sensorVelocityFlag, "sensorVelocityFlag");
		parsePrimative<bool>(true, staggerdVelocityFlag, "staggerdVelocityFlag");

	}
};