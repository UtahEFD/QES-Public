//
//  Sources.hpp
//  
//  This class represents source types
//  this is the source input. Eventually the GUI can help to set up everything,
//   always converting it into some form of a point source using builtin tools.
//
//  Created by Jeremy Gibbs on 03/28/19.
//  Updated by Loren Atwood on 11/09/19.
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
    
    	int numSources;		// number of sources, you fill in source information for each source next
    	std::vector<SourceKind*> sources;		// source type and the collection of all the different sources from input
    
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