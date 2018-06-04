#pragma once

#include "ParseInterface.h"
#include "X.h"

class Root : public ParseInterface
{
public:
	X* xVar;

	void parseValues()
	{
		parseElement<X>(true, xVar, "X");
	}
};