#pragma once

#include "ParseInterface.h"

class FileOptions : public ParseInterface
{
private:
	int outputFormat;
	bool massConservedFlag;
	bool sensorVelocityFlag;
	bool staggerdVelocityFlag;


public:

	virtual void parseValues()
	{
		parsePrimative<int>(true, outputFormat, "outputFormat");
		parsePrimative<bool>(true, massConservedFlag, "massConservedFlag");
		parsePrimative<bool>(true, sensorVelocityFlag, "sensorVelocityFlag");
		parsePrimative<bool>(true, staggerdVelocityFlag, "staggerdVelocityFlag");

	}
};