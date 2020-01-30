//
//  Sources.hpp
//  
//  This class represents source types
//
//  Created by Jeremy Gibbs on 03/28/19.
//

#ifndef SOURCES_HPP
#define SOURCES_HPP


#include "SourceKind.hpp"
#include "SourcePoint.hpp"
#include "SourceLine.hpp"
#include "SourceCircle.hpp"
#include "SourceCube.hpp"
#include "SourceFullDomain.hpp"

#include "util/ParseInterface.h"


class Sources : public ParseInterface
{
    private:
    
    public:
    
    	int numSources;
    	std::vector<SourceKind*> sources;
    
    	virtual void parseValues() {
    		parsePrimitive<int>(true, numSources, "numSources");
			parseMultiPolymorphs(false, sources, Polymorph<SourceKind, SourcePoint>("SourcePoint"));
			parseMultiPolymorphs(false, sources, Polymorph<SourceKind, SourceLine>("SourceLine"));
			parseMultiPolymorphs(false, sources, Polymorph<SourceKind, SourceCircle>("SourceCircle"));
			parseMultiPolymorphs(false, sources, Polymorph<SourceKind, SourceCube>("SourceCube"));
			parseMultiPolymorphs(false, sources, Polymorph<SourceKind, SourceFullDomain>("SourceFullDomain"));
    	}
};
#endif