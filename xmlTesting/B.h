#pragma once

#include "ParseInterface.h"
#include "A.h"

namespace pt = boost::property_tree;

class B : public ParseInterface
{
public:
	int numAs;
	std::vector<A*> aVals;

	B()
	{
		numAs = 0;
		aVals.clear();
	}

	void parseValues(pt::ptree tree)
	{
		parsePrimative<int>(numAs, "numAs", tree);
		parseMultiElements<A>(aVals, "A", tree);
	}
};