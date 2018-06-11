#pragma once

#include "ParseInterface.h"

class Building : public ParseInterface
{
protected:

public:
	int groupID;
	int buildingType;
	virtual void parseValues() = 0;
};