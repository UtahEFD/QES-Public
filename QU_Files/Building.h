#pragma once

#include "ParseInterface.h"

class Building : public ParseInterface
{
protected:
	int groupID;
	int buildingType;
public:
	virtual void parseValues() = 0;
};