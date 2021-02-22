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
    std::vector<std::string> outputFields;
    bool massConservedFlag;
    bool sensorVelocityFlag;
    bool staggerdVelocityFlag;
    
    FileOptions()
    {
        outputFlag = 0;
        massConservedFlag = false;
        sensorVelocityFlag = false;
        staggerdVelocityFlag = false;
    }
    
    virtual void parseValues()
    {
        parsePrimitive<int>(true, outputFlag, "outputFlag");
        parseMultiPrimitives<std::string>(false, outputFields, "outputFields");
        parsePrimitive<bool>(false, massConservedFlag, "massConservedFlag");
        parsePrimitive<bool>(false, sensorVelocityFlag, "sensorVelocityFlag");
        parsePrimitive<bool>(false, staggerdVelocityFlag, "staggerdVelocityFlag");
    }
};
