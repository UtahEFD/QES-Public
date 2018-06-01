#pragma once

#include "ParseInterface.h"
#include "A.h"
#include "B.h"
#include "C.h"

class X : public ParseInterface
{
public:
	A* aVar;
	B* bVar;
	C* cVar;
	int x;

	void parseValues()
	{
					std::cout << "here\n";
		parsePrimative<int>(x, "intVal");
		parseElement<A>(aVar, "A");
		parseElement<B>(bVar, "B");
		parseElement<C>(cVar, "C");
	}
};