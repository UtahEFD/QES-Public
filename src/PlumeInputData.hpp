//
//  PlumeInputData.hpp
//  
//  This class represents all xml settings
//
//  Created by Jeremy Gibbs on 03/28/19.
//

#ifndef PLUMEINPUTDATA_HPP
#define PLUMEINPUTDATA_HPP

#include "util/ParseInterface.h"

#include "SimulationParameters.hpp"
#include "CollectionParameters.hpp"
#include "FileOptions.hpp"
#include "Sources.hpp"

class PlumeInputData : public ParseInterface {
    
    public:
    	SimulationParameters* simParams;
    	CollectionParameters* colParams;
    	FileOptions* fileOptions;
    	Sources* sources;
    
    
    	PlumeInputData() {
    	    fileOptions = 0;
            simParams = 0;
            colParams = 0;
            sources = 0;
    	}
    
    	virtual void parseValues() {
    	    parseElement<SimulationParameters>(true, simParams, "simulationParameters");
    	    parseElement<CollectionParameters>(true, colParams, "collectionParameters");
    	    parseElement<FileOptions>(true, fileOptions, "fileOptions");
    	    parseElement<Sources>(false, sources, "sources");
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