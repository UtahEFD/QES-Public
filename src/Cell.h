#pragma once

#include "Vector3.h"
#include "Edge.h"
#include <vector>

enum cellType : int {air_CT, terrain_CT};

/*
 *This class holds information about cell data. This mostly
 *pertains to what entities exit in the cell, and what coordinate
 *points form the terrain in the cell and what edges connect
 *the points. (if it exists) 
 */
class Cell
{
private:
	bool isAir, isTerrain, isCutCell;
	std::vector< Vector3<float> > terrainPoints;
	std::vector< Edge< int > > terrainEdges;
public:

	/*
	 *returns if air exists in the cell
	 *@return -true if air exists, false if it doesn't
	 */
	bool getIsAir() {return isAir;}

	/*
	 *returns if terrain exists in the cell
	 *@return -true if rettain exists, false if it doesn't
	 */
	bool getIsTerrain() {return isTerrain;}

	/*
	 *returns if terrain is partially in the cell
	 *@return -true if both terrain and air is in the cell, else false
	 */
	bool getIsCutCell() {return isCutCell;}

	/*
	 *returns a list of coordinate points that form the terrain in the cell
	 *@return -the list of points
	 */
	std::vector< Vector3<float> > getTerrainPoints() {return terrainPoints;}

	/*
	 *returns a list of edges that connect the terrain points in the cell
	 *@return -the list of edges
	 */
	std::vector< Edge< int > > getTerrainEdges() {return terrainEdges;}

	/*
	 *Defaults all entity existances values to false, and has no terrain points
	 */
	Cell();

	/*
	 *Takes in a specific entity that totally fills the cell, set's that value
	 *to true.
	 *@param type_CT -what is filling the cell
	 */
	Cell(int type_CT);

	/*
	 *Takes in a list of terrain points that exist in the cell separating where
	 *the terrain and air exist in the cell, sets isCutCell to true.
	 *@param points -a list of points that form the cut.
	 *@param edges -a list of edges that form the terrain
	 */
	Cell( const std::vector< Vector3<float> >& points, const std::vector< Edge< int > >& edges);
};