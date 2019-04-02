#pragma once

#include "Vector3.h"
#include "Edge.h"
#include <vector>

enum cellType : int {air_CT, terrain_CT};
enum cellFace : int {faceXZNeg_CF, faceXZPos_CF, faceYZNeg_CF, faceYZPos_CF, faceXYNeg_CF, faceXYPos_CF };

/*
 *This class holds information about cell data. This mostly
 *pertains to what entities exit in the cell, and what coordinate
 *points form the terrain in the cell and what edges connect
 *the points. (if it exists) . If a cell is a cut cell it will also
 *have a set of points that lie on each face.
 */
class Cell
{
private:
	bool isAir, isTerrain, isCutCell;
	std::vector< Vector3<float> > terrainPoints;
	std::vector< Edge< int > > terrainEdges;
	std::vector< Vector3<float> > fluidFacePoints[6];
	Vector3<float> location;
	Vector3<float> dimensions;
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
	 *@param location -the position of the corner closest to the origin
	 *@param dimensions -the size of the cell in the xyz directions
	 */
	Cell(const int type_CT, Vector3<float> locationN, Vector3<float> dimensionsN);

	/*
	 *Takes in a list of terrain points that exist in the cell separating where
	 *the terrain and air exist in the cell, sets isCutCell to true.
	 *@param points -a list of points that form the cut.
	 *@param edges -a list of edges that form the terrain
	 *@param intermed -a collection of intermediate points between corners that rest on the top and bottom of the cell
	 *@param location -the position of the corner closest to the origin
	 *@param dimensions -the size of the cell in the xyz directions
	 */
	Cell(  std::vector< Vector3<float> >& points,  std::vector< Edge< int > >& edges,  int intermed[4][4][2],
		 Vector3<float> locationN,  Vector3<float> dimensionsN);


	/*
	 *Returns a vector of points that lie on the specified face
	 *@param index -the index of the face to be returned (cellFace enum)
	 */
	std::vector< Vector3<float> > getFaceFluidPoints(const int index) {return fluidFacePoints[index % 6];}


	/*
	 *Returns the xyz location of the cell from the corner closest to the origin
	 *@return -The location of the cell
	 */
	Vector3<float> getLocationPoints() {return location;}

};

