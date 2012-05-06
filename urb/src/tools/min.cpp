#include <iostream>
#include <cstdio>

#include "../util/matrixIO.h"

float hostFindMin(float* M, int size);

int main(int argc, char* argv[]) 
{
		std::cout << std::endl << "<===>Find min of matrix <===>" << std::endl;
		//Arguments passed should be: filename, filename, nx, ny, nz.

	char* matFile1;

	int nx; int ny; int nz;

	if(argc == 2) 
	{
		matFile1 = argv[1];
	}

	else 
	{
		std::cout << "Expected: filename" << std::endl;
	}

		std::cout << "Loading matrix..." << std::flush;
	float* M_1 = inputMatrix(matFile1, &nx, &ny, &nz);
		std::cout << "done." << std::endl;

	float min = 0.0;

		std::cout << "Finding min of " << matFile1 << "..." << std::flush;
	min = hostFindMin(M_1, nx*ny*nz);
		std::cout << "done." << std::endl;

	printf("Min = %3.3f\n", min);

		return 0;	
	}

float hostFindMin(float* M, int size)
{
	float min = M[0];
	for(int i = 1; i < size; i++)
	{
		if(M[i] < min) {min = M[i];}
	}
	return min;
}

