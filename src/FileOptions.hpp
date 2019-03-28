//
//  FileOptions.hpp
//  
//  This class handles xml output options
//
//  Created by Jeremy Gibbs on 03/25/19.
//

#ifndef FILEOPTIONS_HPP
#define FILEOPTIONS_HPP

#include "util/ParseInterface.h"
#include <string>
#include <vector>

class FileOptions : public ParseInterface {
    private:
    
    public:
    
    	int outputFlag;
    	std::vector<std::string> outputFields;
    
    	virtual void parseValues() {
    		parsePrimitive<int>(true, outputFlag, "outputFlag");
    		parseMultiPrimitives<std::string>(false, outputFields, "outputFields");    
    	}
};
#endif