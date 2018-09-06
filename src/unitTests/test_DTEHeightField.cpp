#include "test_DTEHeightField.h"

std::string test_DTEHeightField::mainTest()
{
	DTEHF = DTEHeightField();
	std::string result;
	/**********
	 *cutCells*
	 *********/
	result = testCutCells();

	if (result != TEST_PASS)
		return result;


	return TEST_PASS;
}

#define CELL(i,j,k) ((i) + (j) * (nx) + (k) * (nx) * (ny))
std::string test_DTEHeightField::testCutCells()
{
	//cutCells(Cell* cells, int i, int j, int nx, int ny, int nz, float dz, Vector3<float> corners[])

	Cell* cells;
	cells = new Cell[20]; //4 stacks of 5 cells. 
	int nx = 4, ny = 1, nz = 5;
	float dz = 1.0f; //dx and dy are also 1.0 but that doesn't entirely matter
	Vector3<float> corners[4];
	std::vector<int> cutCells;


	//initial corners set up :top left then clockwise for stack 1.
	corners[0][0] = 0.0f; corners[0][1] = 1.0f;
	corners[1][0] = 1.0f; corners[1][1] = 1.0f; 
	corners[2][0] = 1.0f; corners[2][1] = 0.0f;
	corners[3][0] = 0.0f; corners[3][1] = 0.0f;

	//Cell Stack 1: dim X:0-1 Y:0-1 Z:0-5 ------------------------------------------------------------ TEST WRITTEN
	//Plane coming up from X:0 to X:1
	corners[0][2] = 1.1f;
	corners[3][2] = 1.1f;
	corners[1][2] = 3.5f;
	corners[2][2] = 3.5f;
	DTEHF.setCellPoints(cells, 0, 0, nx, ny, nz, dz, corners, cutCells);

	//increment corners to next cell stack
	for (int i = 0; i < 4; i++)
		corners[i][0] += 1.0f;

	//Cell Stack 2: ------------------------------------------------------------------------------ TEST NOT WRITTEN
	//Plane coming down from X:1 to X:2
	corners[0][2] = 5.0f;
	corners[3][2] = 5.0f;
	corners[1][2] = 0.0f;
	corners[2][2] = 0.0f;
	DTEHF.setCellPoints(cells, 1, 0, nx, ny, nz, dz, corners, cutCells);

	//increment corners to next cell stack
	for (int i = 0; i < 4; i++)
		corners[i][0] += 1.0f;
	
	//Cell Stack 3:------------------------------------------------------------------------------ TEST NOT WRITTEN
	//Non-Planar increase from X:2Y:0 to X:3Y:1
	corners[0][2] = 0.1f;
	corners[3][2] = 2.3f;
	corners[1][2] = 4.1f;
	corners[2][2] = 0.8f;
	DTEHF.setCellPoints(cells, 2, 0, nx, ny, nz, dz, corners, cutCells);

	//increment corners to next cell stack
	for (int i = 0; i < 4; i++)
		corners[i][0] += 1.0f;
	
	//Cell Stack 4:------------------------------------------------------------------------------ TEST NOT WRITTEN
	//Diagonal Peak across Cell
	corners[0][2] = 0.5f;
	corners[3][2] = 4.3f;
	corners[1][2] = 4.3f;
	corners[2][2] = 0.5f;
	DTEHF.setCellPoints(cells, 3, 0, nx, ny, nz, dz, corners, cutCells);

	///////////////////////////////////
	//Check Results
	///////////////////////////////////
	std::vector< Vector3<float> > cellPoints;

	//Base Cell (1st)
	cellPoints = cells[ CELL(0,0,0) ].getTerrainPoints();
	if (cellPoints.size() != 0)
		return util_errorReport("cutCells", 89, 0, cellPoints.size());

	//Top Cell (5th)
	cellPoints = cells[CELL(0,0,4)].getTerrainPoints();
	if (cellPoints.size() != 0)
		return util_errorReport("cutCells", 94, 0, cellPoints.size());

	//2nd Cell
	cellPoints = cells[CELL(0,0,1)].getTerrainPoints();
	std::vector< Vector3<float> >::iterator it;

	if (cellPoints.size() != 5)
		return util_errorReport("cutCells", 101, 5, cellPoints.size());	

	std::vector< Vector3<float> > pointsToCheck;
	std::vector< std::string > messages;
	pointsToCheck.push_back(Vector3< float >(0.0f, 1.0f, 1.1f)); messages.push_back( "Vector3< float >(0.0f, 1.0f, 1.1f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.0f, 0.0f, 1.1f)); messages.push_back( "Vector3< float >(0.0f, 0.0f, 1.1f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.375f, 1.0f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 1.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.375f, 0.0f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 0.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.375f, 0.375f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 0.375f, 2.0f) was not found" );

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 115, messages[i]);
	}

	//4th Cell
	cellPoints = cells[CELL(0,0,3)].getTerrainPoints();
	if (cellPoints.size() != 5)
		return util_errorReport("cutCells", 121, 5, cellPoints.size());	

	pointsToCheck.clear();
	messages.clear();
	pointsToCheck.push_back(Vector3< float >(1.0f, 1.0f, 3.5f)); messages.push_back( "Vector3< float >(1.0f, 1.0f, 3.5f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.0f, 0.0f, 3.5f)); messages.push_back( "Vector3< float >(1.0f, 0.0f, 3.5f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666f, 1.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666ff, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.79166667f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666ff, 0.79166666666ff, 3.0f) was not found" );

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 135, messages[i]);
	}

	//3rd Cell
	cellPoints = cells[CELL(0,0,2)].getTerrainPoints();
	if (cellPoints.size() != 6)
		return util_errorReport("cutCells", 141, 6, cellPoints.size());	

	pointsToCheck.clear();
	messages.clear();
	pointsToCheck.push_back(Vector3< float >(0.375f, 1.0f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 1.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.375f, 0.0f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 0.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.375f, 0.375f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 0.375f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666f, 1.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666ff, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.79166667f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666ff, 0.79166666666ff, 3.0f) was not found" );

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 156, messages[i]);
	}

	return TEST_PASS;
}