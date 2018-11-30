#pragma once

#include "ParseInterface.h"

class Building : public ParseInterface
{
protected:

public:
	int groupID;
	int buildingType, buildingGeometry;
	float x_start;
	float y_start;
	float L;
	float W;
	int i_start, i_end, j_start, j_end, k_end,k_start;
	int i_cut_start, i_cut_end, j_cut_start, j_cut_end, k_cut_end;
	float H; 
	float baseHeight, baseHeightActual; 


	virtual void parseValues() = 0;
};
