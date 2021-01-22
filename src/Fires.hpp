/*
 * This function contains variables that define information
 * necessary for running the fire code.
 */

#ifndef FIRES_HPP
#define FIRES_HPP

#include <string>
#include "util/ParseInterface.h"
#include "Vector3.h"
#include "DTEHeightField.h"


class Fires : public ParseInterface {
    
    private:
    
    public:
    
    	int numFires,fuelType,fieldFlag;
    	float height,baseHeight,xStart,yStart,length,width,courant;
    	
	std::string fuelFile;

    	virtual void parseValues() {
    		parsePrimitive<int>(true,   numFires,   "numFires");
    		parsePrimitive<int>(true,   fuelType,   "fuelType");
    		parsePrimitive<float>(true, height,     "height");
    		parsePrimitive<float>(true, baseHeight, "baseHeight");
    		parsePrimitive<float>(true, xStart,     "xStart");
    		parsePrimitive<float>(true, yStart,     "yStart");
    		parsePrimitive<float>(true, length,     "length");
    		parsePrimitive<float>(true, width,      "width");
    		parsePrimitive<float>(true, courant,    "courant");
			parsePrimitive<int>(true,   fieldFlag,  "fieldFlag");
        	fuelFile = "";
        	parsePrimitive<std::string>(false, fuelFile, "fuelMap");

            
        
    }
};
#endif
