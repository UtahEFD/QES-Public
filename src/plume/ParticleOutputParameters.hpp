#pragma once

/*
 * This class contains data and variables that set flags and
 * settngs read from the xml.
 */

#include "util/ParseInterface.h"
#include <string>
#include <vector>

class ParticleOutputParameters : public ParseInterface
{
private:
    
public:
    
    float outputStartTime=-1.0;
    float outputEndTime=-1.0;
    float outputFrequency;
    std::vector<std::string> outputFields;
    
    virtual void parseValues()
    {
        parsePrimitive<float>(false, outputStartTime, "outputStartTime");
        parsePrimitive<float>(false, outputEndTime, "outputEndTime");
        parsePrimitive<float>(true, outputFrequency, "outputFrequency");
        parseMultiPrimitives<std::string>(false, outputFields, "outputFields");
    }
};
