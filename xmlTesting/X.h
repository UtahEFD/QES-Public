#pragma once

#include "ParseInterface.h"
#include "A.h"
#include "B.h"
#include "C.h"
#include "D.h"
#include "P.h"

class X : public ParseInterface
{
public:
	A* aVar;
	B* bVar;
	C* cVar;
	D* dVar;
	P* pVar;
	int x;

	void parseValues()
	{
		parsePrimative<int>(true, x, "intVal");
		parseElement<A>(true, aVar, "A");
		parseElement<B>(true, bVar, "B");
		parseElement<C>(true, cVar, "C");
		parsePolymorph(true, pVar, Polymorph<P,P1>("P1"), Polymorph<P,P2>("P2"));
		parseElement<D>(true, dVar, "D");
	}
};