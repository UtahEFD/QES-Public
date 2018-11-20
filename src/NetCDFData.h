#pragma once

#include <vector>
#include <netcdf>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "Cell.h"
using namespace netCDF;
using namespace netCDF::exceptions;

class NetCDFData
{
private:

	float *x, *y, *z;
	int dimX, dimY, dimZ;
	double *u, *v, *w;


	int *iCellFlags;
	int dimXF, dimYF, dimZF;
	float *xF, *yF, *zF;
	long size;

	int *cutCellFlags;

public:

	void getData(float* newX, float* newY, float* newZ, double* newU, double* newV, double* newW, int newDX, int newDY, int newDZ);

	void getDataICell(int* newICellFlags, float* newX, float* newY, float* newZ, int nDX, int nDY, int nDZ, long newSize);

	void getCutCellFlags(Cell* cells);

	bool outputICellFlags(std::string fileName);

	bool outputCutCellFlags(std::string fileName);

	bool outputCellFaceResults(std::string fileName);
};