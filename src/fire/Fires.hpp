/*
 * This function contains variables that define information
 * necessary for running the fire code.
 */
#ifndef FIRES_HPP
#define FIRES_HPP

#include <string>
#include "util/ParseInterface.h"
#include "util/Vector3.h"
#include "winds/DTEHeightField.h"
#include "ignition.h"

class Fires : public ParseInterface {
    private:    
    public:    
    	int numFires,fuelType,fieldFlag;
    	float fmc, courant;
		std::vector<ignition*> IG;
		std::string fuelFile;
    	virtual void parseValues() {
    		parsePrimitive<int>(false,   numFires,   "numFires");
    		parsePrimitive<int>(true,   fuelType,   "fuelType");
			parsePrimitive<float>(true, fmc,	"fmc");
    		parsePrimitive<float>(true, courant,    "courant");
			parseMultiElements<ignition>(false, IG, "ignition");
			parsePrimitive<int>(true,   fieldFlag,  "fieldFlag");
        	fuelFile = "";
        	parsePrimitive<std::string>(false, fuelFile, "fuelMap");        
    }
    void parseTree(pt::ptree t)
  	{
  			setTree(t);
  			setParents("root");
  			parseValues();
  	}
};
#endif
