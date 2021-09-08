#include "test_CutCell.h"

std::string test_CutCell::mainTest()
{
	cutCell = Cut_cell();
	std::string result;
	/**********
	 *cutCells*
	 *********/
	result = testCalculateAreaTopBot();

	if (result != TEST_PASS)
		return result;


	return TEST_PASS;
}


/*
void calculateAreaTopBot(std::vector< Vector3<float> > &terrainPoints, 
							 const std::vector< Edge<int> > &terrainEdges, 
							 const int cellIndex, const float dx, const float dy, const float dz, 
							 Vector3 <float> location, std::vector<float> &coef,
							 const bool isBot);
 */
std::string test_CutCell::testCalculateAreaTopBot()
{
	//our imaginary cell is dx,dy,dz = 1,1,1 and is located at the origin
	Vector3<float> location(0.0f, 0.0f, 0.0f);
	float dx, dy, dz;
	dx = dy = dz = 1.0f;
	int cellIndex = 0;
	std::vector<float> coef;
	coef.resize(1);

	//test cut through middle of cell

	//test cut through cell 1/4th of bottom is air, 1/4th of top is terrain

	//test concave shape onto top and bottom 

	//test separated peaks.
}