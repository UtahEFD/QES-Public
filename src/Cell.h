#pragma once

#include "Vector3.h"
#include <vector>

enum cellType : int {air_CT, terrain_CT};

class Cell
{
private:
	bool isAir, isTerrain, isCutCell;
	std::vector< Vector3<float> > terrainPoints;
public:

	bool getIsAir() {return isAir;}
	bool getIsTerrain() {return isTerrain;}
	bool getIsCutCell() {return isCutCell;}
	std::vector< Vector3<float> > getTerrainPoints() {return terrainPoints;}

	Cell();
	Cell(int type_CT);
	Cell( std::vector< Vector3<float> > points);
};