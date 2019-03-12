#pragma once

/*
 * This class contains data and variables that set flags and
 * settngs read from the xml.
 */

#include "util/ParseInterface.h"
#include <string>
#include <vector>

class FileOptions : public ParseInterface
{
private:



public:

	int outputFormat;
	std::vector<std::string> outputFields;
	bool massConservedFlag;
	bool sensorVelocityFlag;
	bool staggerdVelocityFlag;

	virtual void parseValues()
	{
		parsePrimitive<int>(true, outputFormat, "outputFormat");
		parseMultiPrimitives<std::string>(true, outputFields, "outputFields");
		parsePrimitive<bool>(true, massConservedFlag, "massConservedFlag");
		parsePrimitive<bool>(true, sensorVelocityFlag, "sensorVelocityFlag");
		parsePrimitive<bool>(true, staggerdVelocityFlag, "staggerdVelocityFlag");

	}
};
