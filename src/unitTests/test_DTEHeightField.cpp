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

	//Cell Stack 2: ------------------------------------------------------------------------------ TEST WRITTEN
	//Plane coming down from X:1 to X:2
	corners[0][2] = 5.0f;
	corners[3][2] = 5.0f;
	corners[1][2] = 0.0f;
	corners[2][2] = 0.0f;
	DTEHF.setCellPoints(cells, 1, 0, nx, ny, nz, dz, corners, cutCells);

	//increment corners to next cell stack
	for (int i = 0; i < 4; i++)
		corners[i][0] += 1.0f;
	
	//Cell Stack 3:------------------------------------------------------------------------------ TEST WRITTEN
	//Non-Planar increase from X:2Y:0 to X:3Y:1
	corners[0][2] = 0.1f;
	corners[1][2] = 2.3f;
	corners[2][2] = 4.1f;
	corners[3][2] = 0.8f;
	DTEHF.setCellPoints(cells, 2, 0, nx, ny, nz, dz, corners, cutCells);

	//increment corners to next cell stack
	for (int i = 0; i < 4; i++)
		corners[i][0] += 1.0f;
	
	//Cell Stack 4:------------------------------------------------------------------------------ TEST BEING WRITTEN
	//Diagonal Peak across Cell
	corners[0][2] = 4.3f;
	corners[1][2] = 0.5f;
	corners[2][2] = 4.3f;
	corners[3][2] = 0.5f;
	DTEHF.setCellPoints(cells, 3, 0, nx, ny, nz, dz, corners, cutCells);

	///////////////////////////////////
	//Check Results -- Cell Stack 1
	///////////////////////////////////
	std::vector< Vector3<float> > cellPoints;
	std::vector< Edge<int> >cellEdges;

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
	cellEdges = cells[CELL(0,0,1)].getTerrainEdges();
	std::vector< Vector3<float> >::iterator it; std::vector< Edge< int > >::iterator itE;

	if (cellPoints.size() != 7)
		return util_errorReport("cutCells", 103, 7, cellPoints.size());	
	if (cellEdges.size() != 12)
		return util_errorReport("cutCells", 105, 12, cellEdges.size());

	std::vector< Vector3<float> > pointsToCheck;
	std::vector< std::string > messages;
	pointsToCheck.push_back(Vector3< float >(0.0f, 1.0f, 1.1f)); messages.push_back( "Vector3< float >(0.0f, 1.0f, 1.1f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.0f, 0.0f, 1.1f)); messages.push_back( "Vector3< float >(0.0f, 0.0f, 1.1f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.375f, 1.0f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 1.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.375f, 0.0f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 0.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.375f, 0.625f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 0.625f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.0f, 1.0f, 2.0f)); messages.push_back( "Vector3< float >(1.0f, 1.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.0f, 0.0f, 2.0f)); messages.push_back( "Vector3< float >(1.0f, 0.0f, 2.0f) was not found" );

	std::vector< Edge< int > > edgesToCheck;
	std::vector< std::string > messages_E;

	edgesToCheck.push_back(Edge<int>(0,3)); messages_E.push_back("missing Edge 0,3");
	edgesToCheck.push_back(Edge<int>(0,4)); messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(0,5)); messages_E.push_back("missing Edge 0,5");
	edgesToCheck.push_back(Edge<int>(1,2)); messages_E.push_back("missing Edge 1,2");
	edgesToCheck.push_back(Edge<int>(1,4)); messages_E.push_back("missing Edge 1,4");
	edgesToCheck.push_back(Edge<int>(1,5)); messages_E.push_back("missing Edge 1,5");
	edgesToCheck.push_back(Edge<int>(2,5)); messages_E.push_back("missing Edge 2,5");
	edgesToCheck.push_back(Edge<int>(2,6)); messages_E.push_back("missing Edge 2,6");
	edgesToCheck.push_back(Edge<int>(3,5)); messages_E.push_back("missing Edge 3,5");
	edgesToCheck.push_back(Edge<int>(3,6)); messages_E.push_back("missing Edge 3,6");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(5,6)); messages_E.push_back("missing Edge 5,6");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 137, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 144, messages_E[i]);
	}

	//4th Cell
	cellPoints = cells[CELL(0,0,3)].getTerrainPoints();
	cellEdges = cells[CELL(0,0,3)].getTerrainEdges();

	if (cellPoints.size() != 7)
		return util_errorReport("cutCells", 152, 7, cellPoints.size());	
	if (cellEdges.size() != 12)
		return util_errorReport("cutCells", 154, 12, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();
	pointsToCheck.push_back(Vector3< float >(1.0f, 1.0f, 3.5f)); messages.push_back( "Vector3< float >(1.0f, 1.0f, 3.5f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.0f, 0.0f, 3.5f)); messages.push_back( "Vector3< float >(1.0f, 0.0f, 3.5f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666f, 1.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666f, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.20833333f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666f, 0.20833333f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.0f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(0.0f, 1.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.0f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(0.0f, 0.0f, 3.0f) was not found" );

	edgesToCheck.push_back(Edge<int>(0,3)); messages_E.push_back("missing Edge 0,3");
	edgesToCheck.push_back(Edge<int>(0,4)); messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(0,5)); messages_E.push_back("missing Edge 0,5");
	edgesToCheck.push_back(Edge<int>(1,2)); messages_E.push_back("missing Edge 1,2");
	edgesToCheck.push_back(Edge<int>(1,4)); messages_E.push_back("missing Edge 1,4");
	edgesToCheck.push_back(Edge<int>(1,5)); messages_E.push_back("missing Edge 1,5");
	edgesToCheck.push_back(Edge<int>(2,5)); messages_E.push_back("missing Edge 2,5");
	edgesToCheck.push_back(Edge<int>(2,6)); messages_E.push_back("missing Edge 2,6");
	edgesToCheck.push_back(Edge<int>(3,5)); messages_E.push_back("missing Edge 3,5");
	edgesToCheck.push_back(Edge<int>(3,6)); messages_E.push_back("missing Edge 3,6");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(5,6)); messages_E.push_back("missing Edge 5,6");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 185, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 192, messages_E[i]);
	}

	//3rd Cell
	cellPoints = cells[CELL(0,0,2)].getTerrainPoints();
	cellEdges = cells[CELL(0,0,2)].getTerrainEdges();

	if (cellPoints.size() != 10)
		return util_errorReport("cutCells", 200, 10, cellPoints.size());	
	if (cellEdges.size() != 19)
		return util_errorReport("cutCells", 202, 19, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();
	pointsToCheck.push_back(Vector3< float >(0.375f, 1.0f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 1.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.375f, 0.0f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 0.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.375f, 0.625f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 0.625f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666f, 1.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666f, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.20833333f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666f, 0.20833333f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.0f, 0.0f, 2.0f)); messages.push_back( "Vector3< float >(0.0f, 0.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.0f, 1.0f, 2.0f)); messages.push_back( "Vector3< float >(0.0f, 1.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.0f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(1.0f, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.0f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(1.0f, 1.0f, 3.0f) was not found" );


	edgesToCheck.push_back(Edge<int>(0,3)); messages_E.push_back("missing Edge 0,3");
	edgesToCheck.push_back(Edge<int>(0,5)); messages_E.push_back("missing Edge 0,5");
	edgesToCheck.push_back(Edge<int>(0,7)); messages_E.push_back("missing Edge 0,7");
	edgesToCheck.push_back(Edge<int>(1,2)); messages_E.push_back("missing Edge 1,2");
	edgesToCheck.push_back(Edge<int>(1,4)); messages_E.push_back("missing Edge 1,4");
	edgesToCheck.push_back(Edge<int>(1,6)); messages_E.push_back("missing Edge 1,6");
	edgesToCheck.push_back(Edge<int>(2,6)); messages_E.push_back("missing Edge 2,6");
	edgesToCheck.push_back(Edge<int>(2,8)); messages_E.push_back("missing Edge 2,8");
	edgesToCheck.push_back(Edge<int>(3,7)); messages_E.push_back("missing Edge 3,7");
	edgesToCheck.push_back(Edge<int>(3,9)); messages_E.push_back("missing Edge 3,9");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(4,6)); messages_E.push_back("missing Edge 4,6");
	edgesToCheck.push_back(Edge<int>(5,6)); messages_E.push_back("missing Edge 5,6");
	edgesToCheck.push_back(Edge<int>(5,7)); messages_E.push_back("missing Edge 5,7");
	edgesToCheck.push_back(Edge<int>(6,7)); messages_E.push_back("missing Edge 6,7");
	edgesToCheck.push_back(Edge<int>(6,8)); messages_E.push_back("missing Edge 6,8");
	edgesToCheck.push_back(Edge<int>(6,9)); messages_E.push_back("missing Edge 6,9");
	edgesToCheck.push_back(Edge<int>(7,9)); messages_E.push_back("missing Edge 7,9");
	edgesToCheck.push_back(Edge<int>(8,9)); messages_E.push_back("missing Edge 8,9");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 244, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 251, messages_E[i]);
	}

	///////////////////////////////////
	//Check Results -- Cell Stack 2
	///////////////////////////////////

	//note, cells 1 and 3 should have the same edges as 2 so they aren't necessary to check

	//Bottom Cell [0]
	cellPoints = cells[CELL(1,0,0)].getTerrainPoints();
	cellEdges = cells[CELL(1,0,0)].getTerrainEdges();

	if (cellPoints.size() != 7)
		return util_errorReport("cutCells", 266, 7, cellPoints.size());	
	if (cellEdges.size() != 12)
		return util_errorReport("cutCells", 269, 12, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();

	pointsToCheck.push_back(Vector3< float >(1.0f, 1.0f, 1.0f)); messages.push_back( "Vector3< float >(1.0f, 1.0f, 1.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.0f, 0.0f, 1.0f)); messages.push_back( "Vector3< float >(1.0f, 0.0f, 1.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.8f, 1.0f, 1.0f)); messages.push_back( "Vector3< float >(1.8f, 1.0f, 1.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.8f, 0.2f, 1.0f)); messages.push_back( "Vector3< float >(1.2f, 0.8f, 1.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.8f, 0.0f, 1.0f)); messages.push_back( "Vector3< float >(1.8f, 0.0f, 1.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 1.0f, 0.0f)); messages.push_back( "Vector3< float >(2.0f, 1.0f, 0.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 0.0f, 0.0f)); messages.push_back( "Vector3< float >(2.0f, 0.0f, 0.0f) was not found" );


	edgesToCheck.push_back(Edge<int>(0,3)); messages_E.push_back("missing Edge 0,3");
	edgesToCheck.push_back(Edge<int>(0,4)); messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(0,5)); messages_E.push_back("missing Edge 0,5");
	edgesToCheck.push_back(Edge<int>(1,2)); messages_E.push_back("missing Edge 1,2");
	edgesToCheck.push_back(Edge<int>(1,4)); messages_E.push_back("missing Edge 1,4");
	edgesToCheck.push_back(Edge<int>(1,5)); messages_E.push_back("missing Edge 1,5");
	edgesToCheck.push_back(Edge<int>(2,5)); messages_E.push_back("missing Edge 2,5");
	edgesToCheck.push_back(Edge<int>(2,6)); messages_E.push_back("missing Edge 2,6");
	edgesToCheck.push_back(Edge<int>(3,5)); messages_E.push_back("missing Edge 3,5");
	edgesToCheck.push_back(Edge<int>(3,6)); messages_E.push_back("missing Edge 3,6");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(5,6)); messages_E.push_back("missing Edge 5,6");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 301, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 308, messages_E[i]);
	}

	//Mid Cell [2]
	cellPoints = cells[CELL(1,0,2)].getTerrainPoints();
	cellEdges = cells[CELL(1,0,2)].getTerrainEdges();

	if (cellPoints.size() != 10)
		return util_errorReport("cutCells", 316, 10, cellPoints.size());	
	if (cellEdges.size() != 19)
		return util_errorReport("cutCells", 318, 19, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();
	pointsToCheck.push_back(Vector3< float >(1.4f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(1.4f, 1.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.4f, 0.6f, 3.0f)); messages.push_back( "Vector3< float >(1.4f, 0.6f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.4f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(1.4f, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.6f, 1.0f, 2.0f)); messages.push_back( "Vector3< float >(1.6f, 1.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.6f, 0.4f, 2.0f)); messages.push_back( "Vector3< float >(1.6f, 0.4f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.6f, 0.0f, 2.0f)); messages.push_back( "Vector3< float >(1.6f, 0.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.0f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(1.0f, 1.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.0f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(1.0f, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 1.0f, 2.0f)); messages.push_back( "Vector3< float >(2.0f, 1.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 0.0f, 2.0f)); messages.push_back( "Vector3< float >(2.0f, 0.0f, 2.0f) was not found" );


	edgesToCheck.push_back(Edge<int>(0,3)); messages_E.push_back("missing Edge 0,3");
	edgesToCheck.push_back(Edge<int>(0,4)); messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(0,6)); messages_E.push_back("missing Edge 0,6");
	edgesToCheck.push_back(Edge<int>(1,2)); messages_E.push_back("missing Edge 1,2");
	edgesToCheck.push_back(Edge<int>(1,5)); messages_E.push_back("missing Edge 1,5");
	edgesToCheck.push_back(Edge<int>(1,7)); messages_E.push_back("missing Edge 1,7");
	edgesToCheck.push_back(Edge<int>(2,7)); messages_E.push_back("missing Edge 2,7");
	edgesToCheck.push_back(Edge<int>(2,9)); messages_E.push_back("missing Edge 2,9");
	edgesToCheck.push_back(Edge<int>(3,6)); messages_E.push_back("missing Edge 3,6");
	edgesToCheck.push_back(Edge<int>(3,8)); messages_E.push_back("missing Edge 3,8");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(4,6)); messages_E.push_back("missing Edge 4,6");
	edgesToCheck.push_back(Edge<int>(5,6)); messages_E.push_back("missing Edge 5,6");
	edgesToCheck.push_back(Edge<int>(5,7)); messages_E.push_back("missing Edge 5,7");
	edgesToCheck.push_back(Edge<int>(6,7)); messages_E.push_back("missing Edge 6,7");
	edgesToCheck.push_back(Edge<int>(6,8)); messages_E.push_back("missing Edge 6,8");
	edgesToCheck.push_back(Edge<int>(6,9)); messages_E.push_back("missing Edge 6,9");
	edgesToCheck.push_back(Edge<int>(7,9)); messages_E.push_back("missing Edge 7,9");
	edgesToCheck.push_back(Edge<int>(8,9)); messages_E.push_back("missing Edge 8,9");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 360, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 367, messages_E[i]);
	}

	//Top Cell [4]
	cellPoints = cells[CELL(1,0,4)].getTerrainPoints();
	cellEdges = cells[CELL(1,0,4)].getTerrainEdges();

	if (cellPoints.size() != 7)
		return util_errorReport("cutCells", 376, 7, cellPoints.size());	
	if (cellEdges.size() != 12)
		return util_errorReport("cutCells", 378, 12, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();

	pointsToCheck.push_back(Vector3< float >(1.0f, 1.0f, 5.0f)); messages.push_back( "Vector3< float >(1.0f, 1.0f, 5.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.0f, 0.0f, 5.0f)); messages.push_back( "Vector3< float >(1.0f, 0.0f, 5.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.2f, 1.0f, 4.0f)); messages.push_back( "Vector3< float >(1.2f, 1.0f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.2f, 0.8f, 4.0f)); messages.push_back( "Vector3< float >(1.2f, 0.2f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(1.2f, 0.0f, 4.0f)); messages.push_back( "Vector3< float >(1.2f, 0.0f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 1.0f, 4.0f)); messages.push_back( "Vector3< float >(2.0f, 1.0f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 0.0f, 4.0f)); messages.push_back( "Vector3< float >(2.0f, 0.0f, 4.0f) was not found" );


	edgesToCheck.push_back(Edge<int>(0,3)); messages_E.push_back("missing Edge 0,3");
	edgesToCheck.push_back(Edge<int>(0,4)); messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(0,5)); messages_E.push_back("missing Edge 0,5");
	edgesToCheck.push_back(Edge<int>(1,2)); messages_E.push_back("missing Edge 1,2");
	edgesToCheck.push_back(Edge<int>(1,4)); messages_E.push_back("missing Edge 1,4");
	edgesToCheck.push_back(Edge<int>(1,5)); messages_E.push_back("missing Edge 1,5");
	edgesToCheck.push_back(Edge<int>(2,5)); messages_E.push_back("missing Edge 2,5");
	edgesToCheck.push_back(Edge<int>(2,6)); messages_E.push_back("missing Edge 2,6");
	edgesToCheck.push_back(Edge<int>(3,5)); messages_E.push_back("missing Edge 3,5");
	edgesToCheck.push_back(Edge<int>(3,6)); messages_E.push_back("missing Edge 3,6");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(5,6)); messages_E.push_back("missing Edge 5,6");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 411, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 419, messages_E[i]);
	}

	///////////////////////////////////
	//Check Results -- Cell Stack 3
	///////////////////////////////////

	//note, cell 1 isn't necessary to check

	//Bottom Cell [0]
	cellPoints = cells[CELL(2,0,0)].getTerrainPoints();
	cellEdges = cells[CELL(2,0,0)].getTerrainEdges();

	if (cellPoints.size() != 7)
		return util_errorReport("cutCells", 430, 7, cellPoints.size());	
	if (cellEdges.size() != 12)
		return util_errorReport("cutCells", 432, 12, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();

	pointsToCheck.push_back(Vector3< float >(2.0f, 1.0f, 0.1f));       messages.push_back( "Vector3< float >2.0f, 1.0f, 0.1f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 1.0f, 1.0f));       messages.push_back( "Vector3< float >3.0f, 1.0f, 1.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.409091f, 1.0f, 1.0f)); messages.push_back( "Vector3< float >2.409091f, 1.0f, 1.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.225f, 0.775f, 1.0f));   messages.push_back( "Vector3< float >2.225f, 0.775f, 1.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.060606f, 0.0f, 1.0f));  messages.push_back( "Vector3< float >2.060606f, 0.0f, 1.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.0f, 1.0f));       messages.push_back( "Vector3< float >3.0f, 0.0f, 1.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 0.0f, 0.8f));       messages.push_back( "Vector3< float >2.0f, 0.0f, 0.8f)) was not found" );


	edgesToCheck.push_back(Edge<int>(0,4)); messages_E.push_back("missing Edge 0,3");
	edgesToCheck.push_back(Edge<int>(0,5)); messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(0,3)); messages_E.push_back("missing Edge 0,5");
	edgesToCheck.push_back(Edge<int>(1,2)); messages_E.push_back("missing Edge 1,2");
	edgesToCheck.push_back(Edge<int>(1,4)); messages_E.push_back("missing Edge 1,4");
	edgesToCheck.push_back(Edge<int>(1,5)); messages_E.push_back("missing Edge 1,5");
	edgesToCheck.push_back(Edge<int>(2,5)); messages_E.push_back("missing Edge 2,5");
	edgesToCheck.push_back(Edge<int>(2,6)); messages_E.push_back("missing Edge 2,6");
	edgesToCheck.push_back(Edge<int>(3,5)); messages_E.push_back("missing Edge 3,5");
	edgesToCheck.push_back(Edge<int>(3,6)); messages_E.push_back("missing Edge 3,6");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(5,6)); messages_E.push_back("missing Edge 5,6");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 465, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 472, messages_E[i]);
	}


	//Mid Cell [2]
	cellPoints = cells[CELL(2,0,2)].getTerrainPoints();
	cellEdges = cells[CELL(2,0,2)].getTerrainEdges();

	if (cellPoints.size() != 10)
		return util_errorReport("cutCells", 481, 10, cellPoints.size());	
	if (cellEdges.size() != 19)
		return util_errorReport("cutCells", 483, 19, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();
	pointsToCheck.push_back(Vector3< float >(2.863636f, 1.0f, 2.0f));  messages.push_back( "Vector3< float >(2.863636f, 1.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.475f, 0.525f, 2.0f));   messages.push_back( "Vector3< float >(2.475f, 0.525f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.725f, 0.275f, 3.0f));   messages.push_back( "Vector3< float >(2.725f, 0.275f, 3.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.6111111f, 3.0f)); messages.push_back( "Vector3< float >(3.0f, 0.6111111f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.363636f, 0.0f, 2.0f));  messages.push_back( "Vector3< float >(2.363636f, 0.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.666666f, 0.0f, 3.0f));  messages.push_back( "Vector3< float >(2.666666f, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 1.0f, 2.0f)); messages.push_back( "Vector3< float >(2.0f, 1.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 1.0f, 2.3f)); messages.push_back( "Vector3< float >(3.0f, 1.0f, 2.3f) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(3.0f, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 0.0f, 2.0f)); messages.push_back( "Vector3< float >(2.0f, 0.0f, 2.0f) was not found" );


	edgesToCheck.push_back(Edge<int>(0,3)); messages_E.push_back("missing Edge 0,3");
	edgesToCheck.push_back(Edge<int>(0,4)); messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(0,6)); messages_E.push_back("missing Edge 0,6");
	edgesToCheck.push_back(Edge<int>(1,4)); messages_E.push_back("missing Edge 1,4");
	edgesToCheck.push_back(Edge<int>(1,5)); messages_E.push_back("missing Edge 1,5");
	edgesToCheck.push_back(Edge<int>(1,7)); messages_E.push_back("missing Edge 1,7");
	edgesToCheck.push_back(Edge<int>(2,7)); messages_E.push_back("missing Edge 2,7");
	edgesToCheck.push_back(Edge<int>(2,8)); messages_E.push_back("missing Edge 2,8");
	edgesToCheck.push_back(Edge<int>(2,5)); messages_E.push_back("missing Edge 2,5");
	edgesToCheck.push_back(Edge<int>(3,6)); messages_E.push_back("missing Edge 3,6");
	edgesToCheck.push_back(Edge<int>(3,9)); messages_E.push_back("missing Edge 3,9");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(4,6)); messages_E.push_back("missing Edge 4,6");
	edgesToCheck.push_back(Edge<int>(5,6)); messages_E.push_back("missing Edge 5,6");
	edgesToCheck.push_back(Edge<int>(5,7)); messages_E.push_back("missing Edge 5,7");
	edgesToCheck.push_back(Edge<int>(5,9)); messages_E.push_back("missing Edge 5,9");
	edgesToCheck.push_back(Edge<int>(5,8)); messages_E.push_back("missing Edge 5,8");
	edgesToCheck.push_back(Edge<int>(6,9)); messages_E.push_back("missing Edge 6,9");
	edgesToCheck.push_back(Edge<int>(8,9)); messages_E.push_back("missing Edge 8,9");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 525, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 532, messages_E[i]);
	}

	//Mid-Upper Cell [3]
	cellPoints = cells[CELL(2,0,3)].getTerrainPoints();
	cellEdges = cells[CELL(2,0,3)].getTerrainEdges();

	if (cellPoints.size() != 10)
		return util_errorReport("cutCells", 540, 10, cellPoints.size());	
	if (cellEdges.size() != 19)
		return util_errorReport("cutCells", 542, 19, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();
	pointsToCheck.push_back(Vector3< float >(2.725f, 0.275f, 3.0f));   messages.push_back( "Vector3< float >(2.725f, 0.275f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.975f, 0.025f, 4.0f));   messages.push_back( "Vector3< float >(2.975f, 0.025f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.666666f, 0.0f, 3.0f));        messages.push_back( "Vector3< float >(2.666666f, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.969696f, 0.0f, 4.0f));        messages.push_back( "Vector3< float >(2.969696f, 0.0f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.055556f, 4.0f));        messages.push_back( "Vector3< float >(3.0f, 0.055556f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.611111f, 3.0f));        messages.push_back( "Vector3< float >(3.0f, 0.611111f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(2.0f, 1.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(3.0f, 1.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.0f, 4.0f)); messages.push_back( "Vector3< float >(3.0f, 0.0f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(2.0f, 0.0f, 3.0f) was not found" );


	edgesToCheck.push_back(Edge<int>(0,1)); messages_E.push_back("missing Edge 0,1");
	edgesToCheck.push_back(Edge<int>(0,3)); messages_E.push_back("missing Edge 0,3");
	edgesToCheck.push_back(Edge<int>(0,5)); messages_E.push_back("missing Edge 0,5");
	edgesToCheck.push_back(Edge<int>(1,5)); messages_E.push_back("missing Edge 1,5");
	edgesToCheck.push_back(Edge<int>(1,7)); messages_E.push_back("missing Edge 1,7");
	edgesToCheck.push_back(Edge<int>(2,4)); messages_E.push_back("missing Edge 2,4");
	edgesToCheck.push_back(Edge<int>(2,8)); messages_E.push_back("missing Edge 2,8");
	edgesToCheck.push_back(Edge<int>(2,6)); messages_E.push_back("missing Edge 2,6");
	edgesToCheck.push_back(Edge<int>(3,5)); messages_E.push_back("missing Edge 3,5");
	edgesToCheck.push_back(Edge<int>(3,9)); messages_E.push_back("missing Edge 3,9");
	edgesToCheck.push_back(Edge<int>(4,6)); messages_E.push_back("missing Edge 4,6");
	edgesToCheck.push_back(Edge<int>(4,8)); messages_E.push_back("missing Edge 4,8");
	edgesToCheck.push_back(Edge<int>(4,7)); messages_E.push_back("missing Edge 4,7");
	edgesToCheck.push_back(Edge<int>(4,9)); messages_E.push_back("missing Edge 4,9");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(5,9)); messages_E.push_back("missing Edge 5,9");
	edgesToCheck.push_back(Edge<int>(5,7)); messages_E.push_back("missing Edge 5,7");
	edgesToCheck.push_back(Edge<int>(6,7)); messages_E.push_back("missing Edge 6,7");
	edgesToCheck.push_back(Edge<int>(8,9)); messages_E.push_back("missing Edge 8,9");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 584, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 592, messages_E[i]);
	}

	//Top Cell [4]
	cellPoints = cells[CELL(2,0,4)].getTerrainPoints();
	cellEdges = cells[CELL(2,0,4)].getTerrainEdges();

	if (cellPoints.size() != 7)
		return util_errorReport("cutCells", 599, 7, cellPoints.size());	
	if (cellEdges.size() != 12)
		return util_errorReport("cutCells", 601, 12, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();
	pointsToCheck.push_back(Vector3< float >(2.975f, 0.025f, 4.0f));   messages.push_back( "Vector3< float >(2.975f, 0.025f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.969696f, 0.0f, 4.0f));        messages.push_back( "Vector3< float >(2.969696f, 0.0f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.055556f, 4.0f));        messages.push_back( "Vector3< float >(3.0f, 0.055556f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 1.0f, 4.0f)); messages.push_back( "Vector3< float >(2.0f, 1.0f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 1.0f, 4.0f)); messages.push_back( "Vector3< float >(3.0f, 1.0f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.0f, 4.1f)); messages.push_back( "Vector3< float >(3.0f, 0.0f, 4.1f) was not found" );
	pointsToCheck.push_back(Vector3< float >(2.0f, 0.0f, 4.0f)); messages.push_back( "Vector3< float >(2.0f, 0.0f, 4.0f) was not found" );


	edgesToCheck.push_back(Edge<int>(0,1)); messages_E.push_back("missing Edge 0,1");
	edgesToCheck.push_back(Edge<int>(0,3)); messages_E.push_back("missing Edge 0,3");
	edgesToCheck.push_back(Edge<int>(0,4)); messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(1,4)); messages_E.push_back("missing Edge 1,4");
	edgesToCheck.push_back(Edge<int>(1,5)); messages_E.push_back("missing Edge 1,5");
	edgesToCheck.push_back(Edge<int>(2,4)); messages_E.push_back("missing Edge 2,4");
	edgesToCheck.push_back(Edge<int>(2,5)); messages_E.push_back("missing Edge 2,5");
	edgesToCheck.push_back(Edge<int>(2,6)); messages_E.push_back("missing Edge 2,6");
	edgesToCheck.push_back(Edge<int>(3,4)); messages_E.push_back("missing Edge 3,4");
	edgesToCheck.push_back(Edge<int>(3,6)); messages_E.push_back("missing Edge 3,6");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(4,6)); messages_E.push_back("missing Edge 4,6");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 633, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 640, messages_E[i]);
	}

	///////////////////////////////////
	//Check Results -- Cell Stack 4
	///////////////////////////////////

	//note, cell 1 and 3 aren't necessary as they are the same as 2

	//Bottom Cell [0]
	cellPoints = cells[CELL(3,0,0)].getTerrainPoints();
	cellEdges = cells[CELL(3,0,0)].getTerrainEdges();

	if (cellPoints.size() != 8)
		return util_errorReport("cutCells", 654, 8, cellPoints.size());	
	if (cellEdges.size() != 13)
		return util_errorReport("cutCells", 656, 13, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();

	pointsToCheck.push_back(Vector3< float >(3.0f, 1.0f, 1.0f));      messages.push_back( "Vector3< float >(3.0f, 1.0f, 1.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(4.0f, 1.0f, 0.5f));      messages.push_back( "Vector3< float >(4.0f, 1.0f, 0.5f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.868421f, 1.0f, 1.0f)); messages.push_back( "Vector3< float >(3.868421f, 1.0f, 1.0f)  was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.131579f, 1.0f)); messages.push_back( "Vector3< float >(3.0f, 0.131579f, 1.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(4.0f, 0.868421f, 1.0f)); messages.push_back( "Vector3< float >(4.0f, 0.868421f, 1.0f)  was not found" );
	pointsToCheck.push_back(Vector3< float >(3.131579f, 0.0f, 1.0f)); messages.push_back( "Vector3< float >(3.131579f, 0.0f, 1.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(4.0f, 0.0f, 1.0f));       messages.push_back( "Vector3< float >(4.0f, 0.0f, 1.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.0f, 0.5f));       messages.push_back( "Vector3< float >(3.0f, 0.0f, 0.5f)) was not found" );


	edgesToCheck.push_back(Edge<int>(0,4)); messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(0,5)); messages_E.push_back("missing Edge 0,5");
	edgesToCheck.push_back(Edge<int>(1,4)); messages_E.push_back("missing Edge 1,4");
	edgesToCheck.push_back(Edge<int>(1,6)); messages_E.push_back("missing Edge 1,6");
	edgesToCheck.push_back(Edge<int>(2,6)); messages_E.push_back("missing Edge 2,6");
	edgesToCheck.push_back(Edge<int>(2,7)); messages_E.push_back("missing Edge 2,7");
	edgesToCheck.push_back(Edge<int>(3,5)); messages_E.push_back("missing Edge 3,5");
	edgesToCheck.push_back(Edge<int>(3,7)); messages_E.push_back("missing Edge 3,7");
	edgesToCheck.push_back(Edge<int>(4,5)); messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(4,6)); messages_E.push_back("missing Edge 4,6");
	edgesToCheck.push_back(Edge<int>(5,6)); messages_E.push_back("missing Edge 5,6");
	edgesToCheck.push_back(Edge<int>(5,7)); messages_E.push_back("missing Edge 5,7");
	edgesToCheck.push_back(Edge<int>(6,7)); messages_E.push_back("missing Edge 6,7");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 691, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 698, messages_E[i]);
	}


	//Mid Cell [2]
	cellPoints = cells[CELL(3,0,2)].getTerrainPoints();
	cellEdges = cells[CELL(3,0,2)].getTerrainEdges();

	if (cellPoints.size() != 12)
		return util_errorReport("cutCells", 707, 12, cellPoints.size());	
	if (cellEdges.size() != 21)
		return util_errorReport("cutCells", 710, 21, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();
	pointsToCheck.push_back(Vector3< float >(3.0f, 1.0f, 3.0f));      messages.push_back( "Vector3< float >(3.0f, 1.0f, 3.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(4.0f, 1.0f, 2.0f));      messages.push_back( "Vector3< float >(4.0f, 1.0f, 2.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.342105f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(3.342105f, 1.0f, 3.0f)  was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.657895f, 3.0f)); messages.push_back( "Vector3< float >(3.0f, 0.657895f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(4.0f, 0.605263f, 2.0f)); messages.push_back( "Vector3< float >(4.0f, 0.605263f, 2.0f)  was not found" );
	pointsToCheck.push_back(Vector3< float >(3.394737f, 0.0f, 2.0f)); messages.push_back( "Vector3< float >(3.394737f, 0.0f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.605263f, 1.0f, 2.0f)); messages.push_back( "Vector3< float >(3.605263f, 1.0f, 2.0f)  was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.394737f, 2.0f)); messages.push_back( "Vector3< float >(3.0f, 0.394737f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(4.0f, 0.342105f, 3.0f)); messages.push_back( "Vector3< float >(4.0f, 0.342105f, 3.0f)  was not found" );
	pointsToCheck.push_back(Vector3< float >(3.657895f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(3.657895f, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(4.0f, 0.0f, 3.0f));       messages.push_back( "Vector3< float >(4.0f, 0.0f, 3.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.0f, 2.0f));       messages.push_back( "Vector3< float >(3.0f, 0.0f, 2.0f)) was not found" );


	edgesToCheck.push_back(Edge<int>(0,4));  messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(0,6));  messages_E.push_back("missing Edge 0,6");
	edgesToCheck.push_back(Edge<int>(1,5));  messages_E.push_back("missing Edge 1,5");
	edgesToCheck.push_back(Edge<int>(1,9));  messages_E.push_back("missing Edge 1,9");
	edgesToCheck.push_back(Edge<int>(2,8));  messages_E.push_back("missing Edge 2,8");
	edgesToCheck.push_back(Edge<int>(2,10)); messages_E.push_back("missing Edge 2,10");
	edgesToCheck.push_back(Edge<int>(3,7));  messages_E.push_back("missing Edge 3,7");
	edgesToCheck.push_back(Edge<int>(3,11)); messages_E.push_back("missing Edge 3,11");
	edgesToCheck.push_back(Edge<int>(4,5));  messages_E.push_back("missing Edge 4,5");
	edgesToCheck.push_back(Edge<int>(4,6));  messages_E.push_back("missing Edge 4,6");
	edgesToCheck.push_back(Edge<int>(4,8));  messages_E.push_back("missing Edge 4,8");
	edgesToCheck.push_back(Edge<int>(4,9));  messages_E.push_back("missing Edge 4,9");
	edgesToCheck.push_back(Edge<int>(5,9));  messages_E.push_back("missing Edge 5,9");
	edgesToCheck.push_back(Edge<int>(6,8));  messages_E.push_back("missing Edge 6,8");
	edgesToCheck.push_back(Edge<int>(6,10)); messages_E.push_back("missing Edge 6,10");
	edgesToCheck.push_back(Edge<int>(6,7));  messages_E.push_back("missing Edge 6,7");
	edgesToCheck.push_back(Edge<int>(6,11)); messages_E.push_back("missing Edge 6,11");
	edgesToCheck.push_back(Edge<int>(7,11)); messages_E.push_back("missing Edge 7,11");
	edgesToCheck.push_back(Edge<int>(8,10)); messages_E.push_back("missing Edge 8,10");
	edgesToCheck.push_back(Edge<int>(8,9));  messages_E.push_back("missing Edge 8,9");
	edgesToCheck.push_back(Edge<int>(10,11));messages_E.push_back("missing Edge 10,11");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 755, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 762, messages_E[i]);
	}

	//Top Cell [4]
	cellPoints = cells[CELL(3,0,4)].getTerrainPoints();
	cellEdges = cells[CELL(3,0,4)].getTerrainEdges();

	if (cellPoints.size() != 9)
		return util_errorReport("cutCells", 770, 9, cellPoints.size());	
	if (cellEdges.size() != 16)
		return util_errorReport("cutCells", 772, 16, cellEdges.size());

	pointsToCheck.clear();
	edgesToCheck.clear();
	messages_E.clear();
	messages.clear();

	pointsToCheck.push_back(Vector3< float >(3.0f, 1.0f, 4.3f));      messages.push_back( "Vector3< float >(3.0f, 1.0f, 4.3f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(4.0f, 1.0f, 4.0f));      messages.push_back( "Vector3< float >(4.0f, 1.0f, 4.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.078947f, 1.0f, 4.0f)); messages.push_back( "Vector3< float >(3.078947f, 1.0f, 4.0f))  was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.921053f, 4.0f)); messages.push_back( "Vector3< float >(3.0f, 0.921053f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(4.0f, 0.078947f, 4.0f)); messages.push_back( "Vector3< float >(4.0f, 0.078947f, 4.0f)  was not found" );
	pointsToCheck.push_back(Vector3< float >(3.921053f, 0.0f, 4.0f)); messages.push_back( "Vector3< float >(3.921053f, 0.0f, 4.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(4.0f, 0.0f, 4.3f));       messages.push_back( "Vector3< float >(4.0f, 0.0f, 4.3f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.0f, 0.0f, 4.0f));       messages.push_back( "Vector3< float >(3.0f, 0.0f, 4.0f)) was not found" );
	pointsToCheck.push_back(Vector3< float >(3.5f, 0.5f, 4.3f));       messages.push_back( "Vector3< float >(3.5f, 0.5f, 4.3f)) was not found" );


	edgesToCheck.push_back(Edge<int>(0,4)); messages_E.push_back("missing Edge 0,4");
	edgesToCheck.push_back(Edge<int>(0,5)); messages_E.push_back("missing Edge 0,5");
	edgesToCheck.push_back(Edge<int>(0,8)); messages_E.push_back("missing Edge 0,8");
	edgesToCheck.push_back(Edge<int>(1,4)); messages_E.push_back("missing Edge 1,4");
	edgesToCheck.push_back(Edge<int>(1,6)); messages_E.push_back("missing Edge 1,6");
	edgesToCheck.push_back(Edge<int>(2,6)); messages_E.push_back("missing Edge 2,6");
	edgesToCheck.push_back(Edge<int>(2,7)); messages_E.push_back("missing Edge 2,7");
	edgesToCheck.push_back(Edge<int>(2,8)); messages_E.push_back("missing Edge 2,8");
	edgesToCheck.push_back(Edge<int>(3,5)); messages_E.push_back("missing Edge 3,5");
	edgesToCheck.push_back(Edge<int>(3,7)); messages_E.push_back("missing Edge 3,7");
	edgesToCheck.push_back(Edge<int>(4,6)); messages_E.push_back("missing Edge 4,6");
	edgesToCheck.push_back(Edge<int>(4,8)); messages_E.push_back("missing Edge 4,8");
	edgesToCheck.push_back(Edge<int>(5,7)); messages_E.push_back("missing Edge 5,7");
	edgesToCheck.push_back(Edge<int>(5,8)); messages_E.push_back("missing Edge 5,8");
	edgesToCheck.push_back(Edge<int>(6,8)); messages_E.push_back("missing Edge 6,8");
	edgesToCheck.push_back(Edge<int>(7,8)); messages_E.push_back("missing Edge 7,8");

	for (int i = 0; i < pointsToCheck.size(); i++)
	{
		it = find(cellPoints.begin(), cellPoints.end(), pointsToCheck[i]);
		if (it == cellPoints.end())
			return util_errorReport("cutCells", 811, messages[i]);
	}

	for (int i = 0; i < edgesToCheck.size(); i++)
	{
		itE = find(cellEdges.begin(), cellEdges.end(), edgesToCheck[i]);
		if (itE == cellEdges.end())
			return util_errorReport("cutCells", 818, messages_E[i]);
	}

	return TEST_PASS;
}