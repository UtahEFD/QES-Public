//
//  SourcePoint.hpp
//  
//  This class represents a point source
//
//  Created by Jeremy Gibbs on 03/28/19.
//

#include "util/ParseInterface.h"
#include "SourceKind.hpp"

class SourcePoint : public SourceKind
{
    protected:
    
    public:
    
        int numParticles;
    	float posX, posY, posZ, radius; 
    
    	virtual void parseValues() {
        	
            parsePrimitive<int>(true, numParticles, "numParticles");
            parsePrimitive<float>(true, posX, "posX");
            parsePrimitive<float>(true, posY, "posY");
            parsePrimitive<float>(true, posZ, "posZ");
            parsePrimitive<float>(true, radius, "radius");
        }
};