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
    
    	float runTime;
		float timeStep;
		double invarianceTol;
		double C_0;
		int updateFrequency_particleLoop;
		int updateFrequency_timeLoop;
    	        
    	virtual void parseValues() {
    		parsePrimitive< float >(true, runTime, "runTime");
    		parsePrimitive< float >(true, timeStep, "timeStep");
			parsePrimitive< double >(true, invarianceTol, "invarianceTol");
			parsePrimitive< double >(true, C_0, "C_0");
			parsePrimitive< int >(true, updateFrequency_particleLoop, "updateFrequency_particleLoop");
			parsePrimitive< int >(true, updateFrequency_timeLoop, "updateFrequency_timeLoop");
    	}
};
#endif