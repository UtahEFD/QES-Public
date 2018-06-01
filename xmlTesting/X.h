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
		parsePrimative<int>(x, "intVal");
		parseElement<A>(aVar, "A");
		parseElement<B>(bVar, "B");
		parseElement<C>(cVar, "C");
		parsePolymorph(pVar, Polymorph<P,P1>("P1"), Polymorph<P,P2>("P2"));
		parseElement<D>(dVar, "D");
	}
};