//
//  Sources.hpp
//  
//  This class represents source types
//
//  Created by Jeremy Gibbs on 03/28/19.
//

#ifndef SOURCES_HPP
#define SOURCES_HPP

#include "util/ParseInterface.h"
#include "SourceKind.hpp"
#include "SourcePoint.hpp"

class Sources : public ParseInterface
{
    private:
    
    public:
    
    	int numSources;
    	std::vector<SourceKind*> sources;
    
    	virtual void parseValues() {
    		parsePrimitive<int>(true, numSources, "numSources");
    		parseMultiPolymorphs(true, sources, Polymorph<SourceKind, SourcePoint>("pointSource"));
    	}
};
#endif