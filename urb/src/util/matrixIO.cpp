#include "matrixIO.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>


/*********************************************************/
/**                   Matrix Printer                    **/
/*********************************************************/

void printMatrixSample(const float* mat, const int& wM, const int& hM) {
	if(wM <= 17 && hM <= 17) {
		for(int i = 0; i < hM; i++) {
			for(int j = 0; j < wM; j++) {
				std::cout << mat[i * wM + j] << "\t" << std::flush;
				}
			std::cout << std::endl;
			}
		}
	else {
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				std::cout << mat[i * wM + j] << "\t" << std::flush;
				}
			std::cout << " | " << std::flush;
			for(int j = wM - 4; j < wM; j++) {
				std::cout << mat[i * wM + j] << "\t" << std::flush;
				}
			std::cout << std::endl;
			}
			std::cout << "--------------------------------   --------------------------------" << std::endl;

		for(int i = hM - 4; i < hM; i++) {
			for(int j = 0; j < 4; j++) {
				std::cout << mat[i * wM + j] << "\t" << std::flush;
				}
			std::cout << " | " << std::flush;
			for(int j = wM - 4; j < wM; j++) {
				std::cout << mat[i * wM + j] << "\t" << std::flush;
				}
			std::cout << std::endl;
			}
		}
	}

void printMatrixSample(const double* mat, const int& wM, const int& hM) {
	if(wM <= 17 && hM <= 17) {
		for(int i = 0; i < hM; i++) {
			for(int j = 0; j < wM; j++) {
				std::cout << mat[i * wM + j] << "\t" << std::flush;
				}
			std::cout << std::endl;
			}
		}
	else {
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				std::cout << mat[i * wM + j] << "\t" << std::flush;
				}
			std::cout << " | " << std::flush;
			for(int j = wM - 4; j < wM; j++) {
				std::cout << mat[i * wM + j] << "\t" << std::flush;
				}
			std::cout << std::endl;
			}
			std::cout << "--------------------------------   --------------------------------" << std::endl;

		for(int i = hM - 4; i < hM; i++) {
			for(int j = 0; j < 4; j++) {
				std::cout << mat[i * wM + j] << "\t" << std::flush;
				}
			std::cout << " | " << std::flush;
			for(int j = wM - 4; j < wM; j++) {
				std::cout << mat[i * wM + j] << "\t" << std::flush;
				}
			std::cout << std::endl;
			}
		}
	}

void printVectorSample(const float* v, const int& s) {
	for(int i = 0; i < 4; i++) {std::cout << v[i] << std::endl;}
	std::cout << std::endl << "--------" << std::endl << std::endl;
	for(int i = s - 4; i < s; i++) {std::cout << v[i] << std::endl;}
	}

void printVectorSample(const double* v, const int& s) {
	for(int i = 0; i < 4; i++) {std::cout << v[i] << std::endl;}
	std::cout << std::endl << "--------" << std::endl << std::endl;
	for(int i = s - 4; i < s; i++) {std::cout << v[i] << std::endl;}
	}

void outputMatrix(const char* fileName, const double* M, const int& wM, const int& lM, const int& hM) {
	std::fstream output(fileName, std::fstream::out);
	output << wM << " " << lM << " " << hM << std::endl;

	fixed(output);

	for(int k = 0; k < hM; k++) {
		for(int j = 0; j < lM; j++) {
			for(int i = 0; i < wM; i++) {
				output << M[k * lM * wM + j * wM + i] << "  ";
				}
			output << "\n";
			}
		output << "\n\n";
		}
	}

void outputMatrix(const char* fileName, const float* M, const int& wM, const int& lM, const int& hM) {
	std::fstream output(fileName, std::fstream::out);
	output << wM << " " << lM << " " << hM << std::endl;

	fixed(output);

	for(int k = 0; k < hM; k++) {
		for(int j = 0; j < lM; j++) {
			for(int i = 0; i < wM; i++) {
				output << M[k * lM * wM + j * wM + i] << "  ";
				}
			output << "\n";
			}
		output << "\n\n";
		}
	}

void outputMatrix(const char* fileName, const int* M, const int& wM, const int& lM, const int& hM) {
	std::fstream output(fileName, std::fstream::out);
	output << wM << " " << lM << " " << hM << std::endl;

	fixed(output);

	for(int k = 0; k < hM; k++) {
		for(int j = 0; j < lM; j++) {
			for(int i = 0; i < wM; i++) {
				output << M[k * lM * wM + j * wM + i] << "  ";
				}
			output << "\n";
			}
		output << "\n\n";
		}
	}

void outputMatrixSample(const char* fileName, const double* M, const int& nx, const int& ny, const int& nz) {
	std::fstream output(fileName, std::fstream::out);

	char* o = new char[15];

	// h = 0 bottom slice (or top)
	int k = 0;
	output << "Slice " << k << "\n";
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
		output << "\n";
		output << "------------------------------------------------------------";
		output << "   ";
		output << "------------------------------------------------------------";
		output << "\n\n";

	for(int i = ny - 4; i < ny; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
	
	
	k = nz - 1; // top slice (or bottom)
	output << "\n\n";
	output << "Slice " << k << "\n";
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
		output << "\n";
		output << "------------------------------------------------------------";
		output << "   ";
		output << "------------------------------------------------------------";
		output << "\n\n";

	for(int i = ny - 4; i < ny; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
	}
	

void outputMatrixSample(const char* fileName, const float* M, const int& nx, const int& ny, const int& nz) {
	std::fstream output(fileName, std::fstream::out);

	char* o = new char[15];

	// h = 0 bottom slice (or top)
	int k = 0;
	output << "Slice " << k << "\n";
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
		output << "\n";
		output << "------------------------------------------------------------";
		output << "   ";
		output << "------------------------------------------------------------";
		output << "\n\n";

	for(int i = ny - 4; i < ny; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
	
	
	k = nz - 1; // top slice (or bottom)
	output << "\n\n";
	output << "Slice " << k << "\n";
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
		output << "\n";
		output << "------------------------------------------------------------";
		output << "   ";
		output << "------------------------------------------------------------";
		output << "\n\n";

	for(int i = ny - 4; i < ny; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%+#13.8g  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
	}

void outputMatrixSample(const char* fileName, const int* M, const int& nx, const int& ny, const int& nz) {
	std::fstream output(fileName, std::fstream::out);

	char* o = new char[15];

	// h = 0 bottom slice (or top)
	int k = 0;
	output << "Slice " << k << "\n";
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%d  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%d  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
		output << "\n";
		output << "------------------------------------------------------------";
		output << "   ";
		output << "------------------------------------------------------------";
		output << "\n\n";

	for(int i = ny - 4; i < ny; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%d  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%d  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
	
	
	k = nz - 1; // top slice (or bottom)
	output << "\n\n";
	output << "Slice " << k << "\n";
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%d  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%d  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
		output << "\n";
		output << "------------------------------------------------------------";
		output << "   ";
		output << "------------------------------------------------------------";
		output << "\n\n";

	for(int i = ny - 4; i < ny; i++) {
		for(int j = 0; j < 4; j++) {
			sprintf(o, "%d  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << " | ";
		for(int j = nx - 4; j < nx; j++) {
			sprintf(o, "%d  ", M[k * ny * nx + i * nx + j]); output << o;
			}
		output << "\n";
		}
	}


void outputVector(const char* fileName, const double* V, const int& lV) {
	std::fstream output(fileName, std::fstream::out);

	char* o = new char[15];

	for(int i = 0; i < lV; i++) {sprintf(o, "%+#13.8g\n", V[i]); output << o;}
	}

void outputVector(const char* fileName, const float* V, const int& lV) {
	std::fstream output(fileName, std::fstream::out);

	char* o = new char[15];

	for(int i = 0; i < lV; i++) {sprintf(o, "%+#13.8g\n", V[i]); output << o;}
	}

void outputVector(const char* fileName, const int* V, const int& lV) {
	std::fstream output(fileName, std::fstream::out);

	char* o = new char[15];

	for(int i = 0; i < lV; i++) {sprintf(o, "%d\n", V[i]); output << o;}
	}

void outputVectorSample(const char* fileName, const double* V, const int& lV) {
	std::fstream output(fileName, std::fstream::out);

	char* o = new char[15];

	for(int i = 0; i < 4; i++) {sprintf(o, "%+#13.8g\n", V[i]); output << o;}
	output << "\n";
	output << "-------------";
	output << "\n\n";
	for(int i = lV - 4; i < lV; i++) {sprintf(o, "%+#13.8g\n", V[i]); output << o;}
	}

void outputVectorSample(const char* fileName, const float* V, const int& lV) {
	std::fstream output(fileName, std::fstream::out);

	char* o = new char[15];

	for(int i = 0; i < 4; i++) {sprintf(o, "%+#13.8g\n", V[i]); output << o;}
	output << "\n";
	output << "-------------";
	output << "\n\n";
	for(int i = lV - 4; i < lV; i++) {sprintf(o, "%+#13.8g\n", V[i]); output << o;}
	}

void outputVectorSample(const char* fileName, const int* V, const int& lV) {
	std::fstream output(fileName, std::fstream::out);

	char* o = new char[15];

	for(int i = 0; i < 4; i++) {sprintf(o, "%d\n", V[i]); output << o;}
	output << "\n";
	output << "-------------";
	output << "\n\n";
	for(int i = lV - 4; i < lV; i++) {sprintf(o, "%d\n", V[i]); output << o;}
	}

///No working quite right. Returns true when not equal. Something to do with tolerance. Use next one.
bool matricesEqualQ(const float* A, const int& x_a, const int& y_a, const int& z_a, 
					const float* B, const int& x_b, const int& y_b, const int& z_b) 
	{
	if(x_a == x_b && y_a == y_b && z_a == z_b) {
		for(int i = 0; i < x_a * y_a * z_a; i++) {
			if(A[i] - B[i] > 0) {
				std::cout << "\t" << "Equal failed with these values and linear location: " << std::flush;
				std::cout << A[i] << " " << B[i] << " " << i << std::endl;
				return false;
				}
			}
		return true;
		}
	else {return false;}
	}

bool matricesNearlyEqualQ(const float* A, const int& x_a, const int& y_a, const int& z_a, 
						  const float* B, const int& x_b, const int& y_b, const int& z_b, const float& tol) {
	if(x_a == x_b && y_a == y_b && z_a == z_b) {
		/*
		for(int i = 0; i < x_a * y_a * z_a; i++) {
			if(fabs(A[i] - B[i]) > tol) {
				std::cout << "\t" << "NearlyEqual failed with these values, difference and linear location: " << std::flush;
				std::cout << A[i] << " " << B[i] << " " << fabs(A[i] - B[i]) << " " << i << std::endl;
				return false;
				}
			}
		*/
		int slice = y_a * x_a;
		int row   = x_a;
		for(int k = 0; k < z_a; k++) {
			for(int j = 0; j < y_a; j++) {
				for(int i = 0; i < x_a; i++) {
					if(fabs(A[k*slice + j*row + i] - B[k*slice + j*row + i]) > tol) {
						std::cout << "\t" << "NearlyEqual failed with these values, difference and location: " << std::flush;
						std::cout << A[k*slice + j*row + i] << " " << std::flush;
						std::cout << B[k*slice + j*row + i] << " " << std::flush;
						std::cout << fabs(A[k*slice + j*row + i] - B[k*slice + j*row + i]) << " " << std::flush;
						std::cout << "(" << i << ", "<< j << ", " << k << ")" << std::endl;
						return false;
						}
					}
				}
			}
		return true;
		}
	else {return false;}
	}

float matricesAbsoluteDifference
(
	const float* A, const int& x_a, const int& y_a, const int& z_a, 
	const float* B, const int& x_b, const int& y_b, const int& z_b
) {
	float abs_dif = 0.0;
	if(x_a == x_b && y_a == y_b && z_a == z_b) {
		int slice = y_a * x_a;
		int row   = x_a;
		for(int k = 0; k < z_a; k++) {
			for(int j = 0; j < y_a; j++) {
				for(int i = 0; i < x_a; i++) {
					abs_dif += fabs(A[k*slice + j*row + i] - B[k*slice + j*row + i]);
					}
				}
			}
		}
	return abs_dif;
	}

float matricesMaxDifference
(
	const float* A, const int& x_a, const int& y_a, const int& z_a, 
	const float* B, const int& x_b, const int& y_b, const int& z_b,
	int loc[3]
) {
	float max_dif = 0.0;
	loc[0] = 0;
	loc[1] = 0;
	loc[2] = 0;
	float cur_dif = 0.0;
	
	if(x_a == x_b && y_a == y_b && z_a == z_b) {
		int slice = y_a * x_a;
		int row   = x_a;
		for(int k = 0; k < z_a; k++) {
			for(int j = 0; j < y_a; j++) {
				for(int i = 0; i < x_a; i++) {
					cur_dif = fabs(A[k*slice + j*row + i] - B[k*slice + j*row + i]);
					if(cur_dif > max_dif) {
						max_dif = cur_dif;
						loc[0] = i;
						loc[1] = j;
						loc[2] = k;
						}
					}
				}
			}
		}
	return max_dif;
	}

float matrixMax(const float* A, const int& x_a, const int& y_a, const int& z_a) {
	float max = A[0];
	float cur = 0.0;

	for(int i = 0; i < x_a * y_a * z_a; i++) {
		cur = A[i]; if(cur > max) {max = cur;}
		}
	return max;
	}


/*********************************************************/
/**                   Matrix Loader                     **/
/*********************************************************/

/**
*	User is responsible for freeing memory allocated.
*/ 
float* inputMatrix(const char* fileName, int* wM, int* lM, int* hM) {
	FILE* input = fopen(fileName, "r");
	if(input == NULL)
	{
		std::cerr << "Unable to open " << fileName << "." << std::flush;
		return NULL;
	}

	fscanf(input, "%d %d %d", wM, lM, hM);

	//std::cout << *wM << " " << *lM << " " << *hM << std::endl;
	float* M = (float*) malloc((*wM) * (*lM) * (*hM) * sizeof(float));

	for(int k = 0; k < *hM; k++) {
		for(int j = 0; j < *lM; j++) {
			for(int i = 0; i < *wM; i++) {
				fscanf(input, "%g", &M[k * *lM * *wM + j * *wM + i]);
				}
			}
		}
	return M;
	}


/*********************************************************/
/**                   Matrix Copier                     **/
/*********************************************************/

//Prolly a slow way to copy, uh?
void copyDoubleToFloat(float* dst, const double* src, const int& s_w, const int& s_l, const int& s_h) {
	for(int k = 0; k < s_h; k++) {
		for(int j = 0; j < s_l; j++) {
			for(int i = 0; i < s_w; i++) {
				dst[k * s_l * s_w + j * s_w + i] = (float) src[k * s_l * s_w + j * s_w + i];
				}
			}
		}
	}

//Prolly a slow way to copy, uh?
void copyFloatToDouble(double* dst, const float* src, const int& s_w, const int& s_l, const int& s_h) {
	for(int k = 0; k < s_h; k++) {
		for(int j = 0; j < s_l; j++) {
			for(int i = 0; i < s_w; i++) {
				dst[k * s_l * s_w + j * s_w + i] = (double) src[k * s_l * s_w + j * s_w + i];
				}
			}
		}
	}

void copyIntToFloat(float* dst, const int* src, const int& s_w, const int& s_l, const int& s_h) {
	for(int k = 0; k < s_h; k++) {
		for(int j = 0; j < s_l; j++) {
			for(int i = 0; i < s_w; i++) {
				dst[k * s_l * s_w + j * s_w + i] = (float) src[k * s_l * s_w + j * s_w + i];
				}
			}
		}
	}

