#include <iostream>
#include "../util/matrixIO.h"

int main(int argc, char* argv[]) {
		std::cout << std::endl << "<===> Compare two matrices (Equality) <===>" << std::endl;
		std::cout << std::endl << "Doesn't work right. Use matrixCompare, which uses tolerances." << std::endl << std::endl;
		//Arguments passed should be: filename, filename, nx, ny, nz.

	char const* matFile1;
	char const* matFile2;

	int nx; int ny; int nz;

	if(argc == 6) {
		matFile1 = argv[1];
		matFile2 = argv[2];

		nx = atoi(argv[3]);
		ny = atoi(argv[4]);
		nz = atoi(argv[5]);

		//std::cout << matFile1 << " " << matFile2 << " " << nx << " " << ny << " " << nz << std::endl;
		}

	else {
		matFile1 = std::string("sor3d_p1.dat").c_str();
		matFile2 = std::string("p1_fortran.dat").c_str();

		nx = 64;
		ny = 64;
		nz = 21;		
		}

		std::cout << "Loading matrices..." << std::flush;
	float* M_1 = inputMatrix(matFile1, &nx, &ny, &nz);
	float* M_2 = inputMatrix(matFile2, &nx, &ny, &nz);
		std::cout << "done." << std::endl;

		std::cout << "Comparing " << matFile1 << " and " << matFile2 << std::flush;
		std::cout << " with dimensions " << nx << "x" << ny << "x" << nz << "..." << std::endl;		

		bool equal = matricesEqualQ(M_1, nx, ny, nz, M_2, nx, ny, nz);

			std::cout << "\t" << "Element by element, these matrices " << std::flush;
			if(equal) {std::cout << "are " << std::flush;}
			else	  {std::cout << "are NOT " << std::flush;}
			std::cout << "equal to one another" << std::endl << std::endl;
	}

