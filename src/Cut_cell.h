#pragma once

#include <math.h>
#include <algorithm>
#include "Vector3.h"
#include "Edge.h"
#include "Cell.h"
#include "DTEHeightField.h"

class Cut_cell
{
private:

public:

	/*
	 * Assumes DTEHF exists
	 */
	void calculateCoefficient(Cell* cells, const DTEHeightField* DTEHF, int nx, int ny, int nz, float dx, float dy, float dz, 
							 std::vector<float> &n, std::vector<float> &m, std::vector<float> &f, std::vector<float> &e, 
							 std::vector<float> &h, std::vector<float> &g, float pi, std::vector<int> &icellflag);
	
	void reorderPoints(std::vector< Vector3<float>> &cut_points, int index, float pi);

	void sort(std::vector<float> &angle, std::vector< Vector3<float>> &cut_points, float pi);

	void calculateArea(std::vector< Vector3<float>> &cut_points, int cutcell_index, float dx, float dy, float dz, 
					  std::vector<float> &n, std::vector<float> &m, std::vector<float> &f, std::vector<float> &e, 
					  std::vector<float> &h, std::vector<float> &g, int index);

	/*
	 * This function uses the edges that form triangles that lie on either the top or bottom of the cell to 
	 * calculate the terrain area that covers each face.
	 *
	 * @param terrainPoints - the points on the face that mark a separation of terrain and air
	 * @param terrainEdges - a list of edges that exist between terrainPoints
	 * @param cellIndex - the index of the cell (this is needed for the coef)
	 * @param dx - dimension of the cell in the x direction
	 * @param dy - dimension of the cell in the y direction
	 * @param dz - dimension of the cell in the z direction
	 * @param location - the location of the corner of the cell closest to the origin
	 * @param coef - the coefficient that should be updated
	 * @param isBot - states if the area for the bottom or top of the cell should be calculated
	 */
	void calculateAreaTopBot(const std::vector< Vector3<float> > &terrainPoints, 
							 const std::vector< Edge<int> > &terrainEdges, 
							 const int cellIndex, const float dx, const float dy, const float dz, 
							 const Vector3 <float> location, std::vector<float> &coef,
							 const bool isBot);



};
