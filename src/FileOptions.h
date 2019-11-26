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

	int outputFlag;
	bool outputTURBInputFile=false;
	std::vector<std::string> outputFields;
	bool massConservedFlag;
	bool sensorVelocityFlag;
	bool staggerdVelocityFlag;

	virtual void parseValues()
	{
		parsePrimitive<int>(true, outputFlag, "outputFlag");
		parsePrimitive<bool>(false, outputTURBInputFile, "outputTURBInputFile");
		parseMultiPrimitives<std::string>(false, outputFields, "outputFields");
		parsePrimitive<bool>(true, massConservedFlag, "massConservedFlag");
		parsePrimitive<bool>(true, sensorVelocityFlag, "sensorVelocityFlag");
		parsePrimitive<bool>(true, staggerdVelocityFlag, "staggerdVelocityFlag");

	}
};
