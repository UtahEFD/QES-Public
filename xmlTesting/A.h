#pragma once

#include "ParseInterface.h"

class A : public ParseInterface
{
public:

	std::vector<int> vals;

	A()
	{
		vals.clear();
	}

	void parseValues()
	{
		parseMultiPrimatives<int>(true, vals, "AVal");
	}
};