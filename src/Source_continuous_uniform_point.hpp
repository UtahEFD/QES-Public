//
//  SourcePoint.hpp
//  
//  This class represents a point source
//
//  Created by Jeremy Gibbs on 03/28/19.
//

#include "util/ParseInterface.h"
#include "SourceKind.hpp"

class Source_continuous_uniform_point : public SourceKind
{
    protected:
    
    public:

        // initializer
        Source_continuous_uniform_point()
            : SourceKind()
        {
            // What should go here ???  Good to consider this case so we
            // don't have problems down the line.
            
            // My linker is NOT happy without something of these constructor/destructor stuff. Trying to figure out which it is unhappy about
        }

        // destructor
        virtual ~Source_continuous_uniform_point()
        {
        }
    
        int numParticles;
    	float posX, posY, posZ, radius; 
    
    	virtual void parseValues() {
        	
            parsePrimitive<int>(true, numParticles, "numParticles");
            parsePrimitive<float>(true, posX, "posX");
            parsePrimitive<float>(true, posY, "posY");
            parsePrimitive<float>(true, posZ, "posZ");
            parsePrimitive<float>(true, radius, "radius");
        }

        std::vector<particle> outputPointInfo(const double& dt,const double& simDur);


       int emitParticles(const float dt, const float currTime, std::vector<particle> &emittedParticles) {}

};
