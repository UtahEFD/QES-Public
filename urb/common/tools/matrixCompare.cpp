#include <iostream>
#include <cstdlib>
#include "../util/matrixIO.h"

int main(int argc, char* argv[]) {
		std::cout << std::endl << "<===> Compare two matrices <===>" << std::endl;
		//Arguments passed should be: filename, filename, nx, ny, nz.

	char* matFile1;
	char* matFile2;

	int nx, ny, nz;
	int nx_1, ny_1, nz_1;
	int nx_2, ny_2, nz_2;

	if(argc == 3) {
		matFile1 = argv[1];
		matFile2 = argv[2];

		//std::cout << matFile1 << " " << matFile2 << " " << nx << " " << ny << " " << nz << std::endl;
		}

	else {
		std::cout << "Please specify two files (that contain properly formatted data) to compare." << std::endl;
		exit(1);
		}

		std::cout << "Loading matrices..." << std::flush;
	float* M_1 = inputMatrix(matFile1, &nx_1, &ny_1, &nz_1);
	float* M_2 = inputMatrix(matFile2, &nx_2, &ny_2, &nz_2);
		std::cout << "done." << std::endl;

	if(nx_1 != nx_2 || ny_1 != ny_2 || nz_1 != nz_2) {
		std::cout << "Matrices for comparison do not have the same dimensions. Exiting...\n" << std::endl;
		exit(1);
		}
	else {
		nx = nx_1; ny = ny_1; nz = nz_1;
		}

		std::cout << "Comparing " << matFile1 << " and " << matFile2 << std::flush;
		std::cout << " with dimensions " << nx << "x" << ny << "x" << nz << "..." << std::endl;		

	for(float tolerance = .1; tolerance > .00000001; tolerance /= 10) {
		bool nearlyEqual = matricesNearlyEqualQ(M_1, nx, ny, nz, M_2, nx, ny, nz, tolerance);

			std::cout << "\t" << "Element by element, these matrices " << std::flush;
			if(nearlyEqual) {std::cout << "are " << std::flush;}
			else			{std::cout << "are NOT " << std::flush;}
			std::cout << "within " << tolerance << " of one another." << std::endl << std::endl;
		}

	free(M_1);
	free(M_2);

	return 0;
	}

