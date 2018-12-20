#pragma once

#include <math.h>
#include "Vector3.h"
#include "Cell.h"
#include "DTEHeightField.h"

class Cut_cell
{
private:

public:

	void calculateCoefficient(Cell* cells, DTEHeightField* DTEHF, int nx, int ny, int nz, float dx, float dy, float dz, 
							 std::vector<float> &n, std::vector<float> &m, std::vector<float> &f, std::vector<float> &e, 
							 std::vector<float> &h, std::vector<float> &g, float pi, std::vector<int> &icellflag);
	
	void reorderPoints(std::vector< Vector3<float>> &cut_points, int index, float pi);

	void sort(std::vector<float> &angle, std::vector< Vector3<float>> &cut_points, float pi);

	void calculateArea(std::vector< Vector3<float>> &cut_points, int cutcell_index, float dx, float dy, float dz, 
					  std::vector<float> &n, std::vector<float> &m, std::vector<float> &f, std::vector<float> &e, 
					  std::vector<float> &h, std::vector<float> &g, int index);


};
