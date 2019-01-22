#pragma once

/*
 * This class serves as a collection of data to be output in netCDF format along with
 * functions that output the data in nc files.
 */

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

	/*
	 * This function takes in data for outputting velocities with cell face data points.
	 *
	 * @param newX -list of values in the X dimension
	 * @param newY -list of values in the Y dimension
	 * @param newZ -list of values in the Z dimension
	 * @param newU -list of X velocity values in each cell
	 * @param newV -list of Y velocity values in each cell
	 * @param newW -list of Z velocity values in each cell
	 * @param newDX -number of values in the x dimension
	 * @param newDY -number of values in the y dimension
	 * @param newDZ -number of values in the z dimension
	 */
	void getDataFace(float* newX, float* newY, float* newZ, double* newU, double* newV, double* newW, int newDX, int newDY, int newDZ);


	/*
	 * This function takes in data for outputting velocities with cell centered data points.
	 *
	 * @param newX -list of values in the X dimension
	 * @param newY -list of values in the Y dimension
	 * @param newZ -list of values in the Z dimension
	 * @param newU -list of X velocity values in each cell
	 * @param newV -list of Y velocity values in each cell
	 * @param newW -list of Z velocity values in each cell
	 * @param newDX -number of values in the x dimension
	 * @param newDY -number of values in the y dimension
	 * @param newDZ -number of values in the z dimension
	 */
	void getDataCenter(float* newX, float* newY, float* newZ, double* newU, double* newV, double* newW, int newDX, int newDY, int newDZ);

	/*
	 * This function takes in data for outputting all iCellFlag values
	 *
	 * @param newICellFlags -list of all iCellFlag values in each cell
	 * @param newX -list of values in the X dimension
	 * @param newY -list of values in the Y dimension
	 * @param newZ -list of values in the Z dimension
	 * @param newDX -number of values in the x dimension
	 * @param newDY -number of values in the y dimension
	 * @param newDZ -number of values in the z dimension
	 * @param newSize -total number of cells
	 */
	void getDataICell(int* newICellFlags, float* newX, float* newY, float* newZ, int nDX, int nDY, int nDZ, long newSize);

	/*
	 * Outputs ICellFlag values in netcdf format.
	 *
	 * @param fileName -the file to output the data to
	 * @return returns false if creation fails
	 */
	bool outputICellFlags(std::string fileName);

	/*
	 * Outputs velocity values in netcdf format.
	 *
	 * @param fileName -the file to output the data to
	 * @return returns false if creation fails
	 */
	bool outputCellFaceResults(std::string fileName);

	/*
	 * Subtracts all iCellFlags from compare and then Outputs ICellFlag 
	 * values in netcdf format.
	 *
	 * @param compare -the netCDF variable that should be differenced from this variable
	 * @param fileName -the file to output the data to
	 * @return returns false if creation fails
	 */
	bool outputICellFlagsDifference(const NetCDFData* compare, std::string fileName);

	/*
	 * Subtracts all velocities from compare and then Outputs ICellFlag 
	 * values in netcdf format.
	 *
	 * @param compare -the netCDF variable that should be differenced from this variable
	 * @param fileName -the file to output the data to
	 * @return returns false if creation fails
	 */
	bool outputCellResultsDifference( const NetCDFData* compare, std::string fileName);

	void getCutCellFlags(Cell* cells);

	bool outputCutCellFlags(std::string fileName);
};
	
