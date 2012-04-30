#include <iostream>
#include "../util/matrixIO.h"

float hostFindMax(float* M, int size);

int main(int argc, char* argv[]) 
{
		std::cout << std::endl << "<===>Find max of matrix <===>" << std::endl;
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

	float max = 0.0;

		std::cout << "Finding max of " << matFile1 << "..." << std::flush;
	max = hostFindMax(M_1, nx*ny*nz);
		std::cout << "done." << std::endl;

	printf("Max = %3.3f\n", max);

		return 0;	
	}

float hostFindMax(float* M, int size)
{
	float max = M[0];
	for(int i = 1; i < size; i++)
	{
		if(M[i] > max) {max = M[i];}
	}
	return max;
}

