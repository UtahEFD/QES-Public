#pragma once

#include "ParseInterface.h"
#include "A.h"
#include "B.h"
#include "C.h"

namespace pt = boost::property_tree;

class X : public ParseInterface
{
public:
	A* aVar;
	B* bVar;
	C* cVar;
	int x;

	void parseValues(pt::ptree tree)
	{
		parsePrimative<int>(x, "intVal", tree);
		parseElement<A>(aVar, "A", tree);
		parseElement<B>(bVar, "B", tree);
		parseElement<C>(cVar, "C", tree);
	}
};