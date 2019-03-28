//
//  FileOptions.hpp
//  
//  This class rhandles xml collection box options
//
//  Created by Jeremy Gibbs on 03/25/19.
//

#ifndef COLLECTIONPARAMETERS_HPP
#define COLLECTIONPARAMETERS_HPP

#include "util/ParseInterface.h"
#include <string>

class CollectionParameters : public ParseInterface {
    
    private:
    
    public:
    
        int nBoxesX, nBoxesY, nBoxesZ;
        float timeStart, timeEnd, timeAvg ;
    	        
    	virtual void parseValues() {
    		parsePrimitive< float >(true, timeStart, "timeStart");
    		parsePrimitive< float >(true, timeEnd, "timeEnd");
    		parsePrimitive< float >(true, timeAvg, "timeAvg");
    		parsePrimitive< int >(true, nBoxesX, "nBoxesX");
    		parsePrimitive< int >(true, nBoxesY, "nBoxesY");
    		parsePrimitive< int >(true, nBoxesZ, "nBoxesZ");
    	}
};
#endif