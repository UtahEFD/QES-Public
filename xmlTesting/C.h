#pragma once

#include "ParseInterface.h"

namespace pt = boost::property_tree;

class C : public ParseInterface
{
public:
	float y;
	int x;

	C()
	{
		y = 0;
		x = 0;
	}

	void parseValues(pt::ptree tree)
	{
		parsePrimative<int>(x, "intValx", tree);
		parsePrimative<float>(y, "floatValy", tree);
	}
};