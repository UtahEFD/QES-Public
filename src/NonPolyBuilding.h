#pragma once

#include "Building.h"


class NonPolyBuilding : public Building
{
private:


public:
<<<<<<< HEAD
	float x_start;
	float y_start;
	float L;
	float W;
	int i_start, i_end, j_start, j_end, k_end;
	int i_cut_start, i_cut_end, j_cut_start, j_cut_end, k_cut_end;
	float H; 


	virtual void parseValues() = 0;
};
=======
	float xFo;
	float yFo;
	float length;
	float width;

	virtual void parseValues() = 0;
};
>>>>>>> origin/doxygenAdd
