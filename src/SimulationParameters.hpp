//
//  SimulationParamters.hpp
//  
//  This class rhandles xml simulation options
//
//  Created by Jeremy Gibbs on 03/25/19.
//

#ifndef SIMULATIONPARAMETERS_HPP
#define SIMULATIONPARAMETERS_HPP

#include "util/ParseInterface.h"
#include <string>

class SimulationParameters : public ParseInterface {
    
    private:
    
    public:
    
    	float runTime, timeStep;
    	        
    	virtual void parseValues() {
    		parsePrimitive< float >(true, runTime, "runTime");
    		parsePrimitive< float >(true, timeStep, "timeStep");
    	}
};
#endif