#pragma once

#include <math.h>
#include <algorithm>
#include "Vector3.h"
#include "Edge.h"
#include "Cell.h"
#include "DTEHeightField.h"

class WINDSInputData;
class WINDSGeneralData;
/**
* This class basically designed to store and handle information related to the
* cut-cells.
*/
class Cut_cell
{
private:

	const float pi = 4.0f * atan(1.0);

public:

	friend class test_CutCell;

	/**
	 * This function calculates area fraction coefficients used in the cut-cell method.
	 * It takes in a pointer to cell and terrain information (intersection points) and after sorting them for each
	 * face of the cell and calculating coefficients, it sets them to the related solver coefficient (e,f,g,h,m,n)
	 */
	void calculateCoefficient(Cell* cells, const DTEHeightField* DTEHF, const WINDSInputData *WID, WINDSGeneralData *WGD);

  /**
	* This function takes in intersection points for each face and reorder them based on angle. It
	* fisrt calculates the centroid of points (simple average). Then it reorders points based on their angle
	* start from -180 to 180 degree.
	*/
	void reorderPoints(std::vector< Vector3<float>> &cut_points, int index, float pi);

	/**
	* This function takes in points and their calculated angles and sort them from lowest to
	* largest.
	*/

	void mergeSort(std::vector<float> &angle, std::vector< Vector3<float>> &cutPoints);

private:

	/**
	* This function takes in sorted intersection points and calculates area fraction coefficients
	* based on polygon area formulation. Then it sets them to related solver coefficients.
	*/
	float calculateArea(std::vector< Vector3<float>> &cut_points, int cutcell_index, float dx, float dy, float dz,
					  std::vector<float> &n, std::vector<float> &m, std::vector<float> &f, std::vector<float> &e,
					  std::vector<float> &h, std::vector<float> &g, int index);

	/*1
	 * This function uses the edges that form triangles that lie on either the top or bottom of the cell to
	 * calculate the terrain area that covers each face.
	 *
	 * @param terrainPoints - the points in the cell that mark a separation of terrain and air
	 * @param terrainEdges - a list of edges that exist between terrainPoints
	 * @param cellIndex - the index of the cell (this is needed for the coef)
	 * @param dx - dimension of the cell in the x direction
	 * @param dy - dimension of the cell in the y direction
	 * @param dz - dimension of the cell in the z direction
	 * @param location - the location of the corner of the cell closest to the origin
	 * @param coef - the coefficient that should be updated
	 * @param isBot - states if the area for the bottom or top of the cell should be calculated
	 */
	float calculateAreaTopBot(std::vector< Vector3<float> > &terrainPoints,
							 const std::vector< Edge<int> > &terrainEdges,
							 const int cellIndex, const float dx, const float dy, const float dz,
							 Vector3 <float> location, std::vector<float> &coef,
							 const bool isBot);



};
