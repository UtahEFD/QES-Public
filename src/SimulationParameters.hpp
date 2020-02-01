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
    
    	float simDur;		// this is the amount of time to run the simulation, lets you have an arbitrary start time to an arbitrary end time
		float timeStep;			// this is the overall integration timestep
		double invarianceTol;		// this is the tolerance used to determine whether makeRealizeable should be run on the stress tensor for a particle
		double C_0;				// this is used to separate out CoEps into its separate parts when doing debug output
		int updateFrequency_particleLoop;		// this is used to know how frequently to print out information during the particle loop of the solver
		int updateFrequency_timeLoop;		// this is used to know how frequently to print out information during the time integration loop of the solver
    	        
    	virtual void parseValues() {
    		parsePrimitive< float >(true, simDur, "simDur");
    		parsePrimitive< float >(true, timeStep, "timeStep");
			parsePrimitive< double >(true, invarianceTol, "invarianceTol");
			parsePrimitive< double >(true, C_0, "C_0");
			parsePrimitive< int >(true, updateFrequency_particleLoop, "updateFrequency_particleLoop");
			parsePrimitive< int >(true, updateFrequency_timeLoop, "updateFrequency_timeLoop");
    	}
};
#endif