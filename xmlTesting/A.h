#pragma once

#include "ParseInterface.h"

namespace pt = boost::property_tree;

class A : public ParseInterface
{
public:

	std::vector<int> vals;

	A()
	{
		vals.clear();
	}

	void parseValues(pt::ptree tree)
	{
		parseMultiPrimatives<int>(vals, "AVal", tree);
	}
};