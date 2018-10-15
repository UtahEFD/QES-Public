#pragma once

#include "ParseInterface.h"

<<<<<<< HEAD
=======
/*
 *Placeholder class for parsed building info in the xml
 */
>>>>>>> origin/doxygenAdd
class Building : public ParseInterface
{
protected:

public:
	int groupID;
	int buildingType, buildingGeometry;
<<<<<<< HEAD
	float H;
	float baseHeight, baseHeightActual; //zfo
	int i_Start, i_end, j_start, j_end, k_start, k_end;
	int i_cut_start, i_cut_end, j_cut_start, j_cut_end, k_cut_end;

	virtual void parseValues() = 0;
};
=======
	float height;
	float baseHeight, baseHeightActual; //zfo
	float centroidX;
	float centroidY;
	int buildingDamage = 0;
	float atten = 0;
	float rotation = 0;
	float Lf, Lr, Weff, Leff, Wt, Lt;
	int iStart, iEnd, jStart, jEnd, kStart, kEnd;
	int buildingRoof = 0;

	virtual void parseValues() = 0;
};
>>>>>>> origin/doxygenAdd
