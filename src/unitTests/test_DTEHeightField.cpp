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
	pointsToCheck.push_back(Vector3< float >(0.375f, 0.375f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 0.375f, 2.0f) was not found" );
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
			return util_errorReport("cutCells", 144, messages[i]);
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
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666ff, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.79166667f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666ff, 0.79166666666ff, 3.0f) was not found" );
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
			return util_errorReport("cutCells", 192, messages[i]);
	}

	//3rd Cell
	cellPoints = cells[CELL(0,0,2)].getTerrainPoints();
	cellEdges = cells[CELL(0,0,3)].getTerrainEdges();

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
	pointsToCheck.push_back(Vector3< float >(0.375f, 0.375f, 2.0f)); messages.push_back( "Vector3< float >(0.375f, 0.375f, 2.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 1.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666f, 1.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.0f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666ff, 0.0f, 3.0f) was not found" );
	pointsToCheck.push_back(Vector3< float >(0.79166667f, 0.79166667f, 3.0f)); messages.push_back( "Vector3< float >(0.79166666666ff, 0.79166666666ff, 3.0f) was not found" );
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
			return util_errorReport("cutCells", 251, messages[i]);
	}

	return TEST_PASS;
}