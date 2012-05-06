#include <iostream>
#include <cstdlib>
#include "../util/matrixIO.h"

int main(int argc, char* argv[]) {
		std::cout << std::endl << "<===> Find Absolute Difference of Two Matrices <===>" << std::endl;
		//Arguments passed should be: filename, filename, nx, ny, nz.

	char const* matFile1;
	char const* matFile2;

	int nx, ny, nz;
	int nx_1, ny_1, nz_1;
	int nx_2, ny_2, nz_2;

	if(argc == 3) {
		matFile1 = argv[1];
		matFile2 = argv[2];

		//std::cout << matFile1 << " " << matFile2 << " " << nx << " " << ny << " " << nz << std::endl;
		}

	else {
		std::cerr << "Please specify two files (that contain properly formatted data) to compare." << std::endl;
		exit(1);
		}

		std::cout << "Loading matrices..." << std::flush;
	float* M_1 = inputMatrix(matFile1, &nx_1, &ny_1, &nz_1);
	float* M_2 = inputMatrix(matFile2, &nx_2, &ny_2, &nz_2);
		std::cout << "done." << std::endl;

	if(M_1 == NULL || M_2 == NULL)
	{
		std::cerr << "Error opening files. Exiting..." << std::endl;
		exit(1);
	}

	if(nx_1 != nx_2 || ny_1 != ny_2 || nz_1 != nz_2) {
		std::cerr << "Matrices for comparison do not have the same dimensions. Exiting...\n" << std::endl;
		exit(1);
		}
	else {
		nx = nx_1; ny = ny_1; nz = nz_1;
		}

		std::cout << "Comparing " << matFile1 << " and " << matFile2 << std::flush;
		std::cout << " with dimensions " << nx << "x" << ny << "x" << nz << "..." << std::endl;		

	
	std::cout << std::fixed;
	std::cout.precision(5);
	
	float absolute_difference = matricesAbsoluteDifference(M_1, nx, ny, nz, M_2, nx, ny, nz);

	std::cout << "  Abs Diff = " << absolute_difference << std::endl;

	int loc[3] = {0,0,0};
	float max_difference = matricesMaxDifference(M_1, nx, ny, nz, M_2, nx, ny, nz, loc);

	std::cout << "  Max Diff = " << max_difference << " at " << std::flush;
	std::cout << "(" << loc[0] << ", " << loc[1] << ", " << loc[2] << ")" << std::endl;

	float avg_difference = absolute_difference / float(nx*ny*nz);
	std::cout << "  Avg Diff = " << avg_difference << std::endl;

	std::cout << std::endl;

	free(M_1);
	free(M_2);
	return 0;
	}

