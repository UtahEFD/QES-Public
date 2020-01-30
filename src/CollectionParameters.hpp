//
//  CollectionParamters.hpp
//  
//  This class handles xml collection box options
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
        float boxBoundsX1, boxBoundsY1, boxBoundsZ1;
        float boxBoundsX2, boxBoundsY2, boxBoundsZ2;
        float timeStart, timeEnd, timeAvg ;
    	        
    	virtual void parseValues() {
        	parsePrimitive< float >(true, timeStart,   "timeStart");
    		parsePrimitive< float >(true, timeEnd,     "timeEnd");
    		parsePrimitive< float >(true, timeAvg,     "timeAvg");
        	parsePrimitive< float >(true, boxBoundsX1, "boxBoundsX1");
    		parsePrimitive< float >(true, boxBoundsY1, "boxBoundsY1");
    		parsePrimitive< float >(true, boxBoundsZ1, "boxBoundsZ1");
    		parsePrimitive< float >(true, boxBoundsX2, "boxBoundsX2");
    		parsePrimitive< float >(true, boxBoundsY2, "boxBoundsY2");
    		parsePrimitive< float >(true, boxBoundsZ2, "boxBoundsZ2");
    		parsePrimitive< int   >(true, nBoxesX,     "nBoxesX");
    		parsePrimitive< int   >(true, nBoxesY,     "nBoxesY");
    		parsePrimitive< int   >(true, nBoxesZ,     "nBoxesZ");
    	}
};
#endif