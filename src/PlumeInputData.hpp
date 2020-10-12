//
//  PlumeInputData.hpp
//  
//  This class represents all xml settings
//
//  Created by Jeremy Gibbs on 03/28/19.
//

#ifndef PLUMEINPUTDATA_HPP
#define PLUMEINPUTDATA_HPP


#include "SimulationParameters.hpp"
#include "CollectionParameters.hpp"
#include "ParticleOutputParameters.hpp"
#include "Sources.hpp"
#include "BoundaryConditions.hpp"


#include "util/ParseInterface.h"


class PlumeInputData : public ParseInterface {
    
public:
    SimulationParameters* simParams;
    CollectionParameters* colParams;
    ParticleOutputParameters* partOutParams;
    Sources* sources;
    BoundaryConditions* BCs;
    
    
    PlumeInputData() {
        simParams = 0;
        colParams = 0;
        partOutParams = 0; 
        sources = 0;
    }
    
    virtual void parseValues() {
        parseElement<SimulationParameters>(true, simParams, "simulationParameters");
        parseElement<CollectionParameters>(true, colParams, "collectionParameters");
        parseElement<ParticleOutputParameters>(false, partOutParams, "particleOutputParameters");

        parseElement<Sources>(false, sources, "sources");
        parseElement<BoundaryConditions>(true, BCs, "boundaryConditions");
        
    }
    /**
     * This function takes in an URBInputData variable and uses it
     * as the base to parse the ptree
     * @param UID the object that will serve as the base level of the xml parser
     */
    void parseTree(pt::ptree t) { //  URBInputData*& UID) {
        // root = new URBInputData();
        setTree(t);
        setParents("root");
        parseValues();
    }
    
    
};
#endif
