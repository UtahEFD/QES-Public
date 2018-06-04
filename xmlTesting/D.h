#pragma once

#include "ParseInterface.h"
#include "P.h"

class D : public ParseInterface
{
public:
	std::vector<P*> pVars;

	D()
	{
		pVars.clear();
	}

	void parseValues()
	{
		parseMultiPolymorphs(true, pVars, Polymorph<P,P1>("P1"), Polymorph<P,P2>("P2"));
	}
};