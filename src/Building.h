#pragma once

#include "ParseInterface.h"

class Building : public ParseInterface
{
protected:

public:
	int groupID;
	int buildingType, buildingGeometry;
	float H;
	float baseHeight, baseHeightActual; //zfo
	int i_Start, i_end, j_start, j_end, k_start, k_end;
	int i_cut_start, i_cut_end, j_cut_start, j_cut_end, k_cut_end;

	virtual void parseValues() = 0;
};
