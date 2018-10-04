#pragma once

#include "ParseInterface.h"
#include "A.h"

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

	void parseValues()
	{
		parsePrimative<int>(true, numAs, "numAs");
		parseMultiElements<A>(true, aVals, "A");
	}
};