/*
* Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Basic C style in and out to files.
*
* Comments: Orignally more C style, but little parts are now C++. All of it
* 					should be C++. Not a great set of functions, but they get the job
*						done. Templates anyone?
*/

#ifndef MATRIXIO_H
#define MATRIXIO_H 1

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>

void printMatrixSample(const float*, int&, int&);
void printMatrixSample(const double*, int&, int&);
void printVectorSample(const float*, int&);
void printVectorSample(const double*, int&);

//Use a void* ???
void outputMatrix(const char* fileName, const double* M, const int& wM, const int& lM, const int& hM);
void outputMatrix(const char* fileName, const float* M, const int& wM, const int& lM, const int& hM);
void outputMatrix(const char* fileName, const int* M, const int& wM, const int& lM, const int& hM);

//Use a void* ???
void outputMatrixSample(const char* fileName, const double* M, const int& wM, const int& lM, const int& hM);
void outputMatrixSample(const char* fileName, const float* M, const int& wM, const int& lM, const int& hM);
void outputMatrixSample(const char* fileName, const int* M, const int& wM, const int& lM, const int& hM);

void outputVector(const char* fileName, const double* V, const int& lV);
void outputVector(const char* fileName, const float* V, const int& lV);
void outputVector(const char* fileName, const int* V, const int& lV);

void outputVectorSample(const char* fileName, const double* V, const int& lV);
void outputVectorSample(const char* fileName, const float* V, const int& lV);
void outputVectorSample(const char* fileName, const int* V, const int& lV);

bool matricesEqualQ
(
	const float* A, const int& x_a, const int& y_a, const int& z_a, 
	const float* B, const int& x_b, const int& y_b, const int& z_b
);
bool matricesNearlyEqualQ
(
	const float* A, const int& x_a, const int& y_a, const int& z_a, 
	const float* B, const int& x_b, const int& y_b, const int& z_b, 
	const float& tol
);
float matricesAbsoluteDifference
(
	const float* A, const int& x_a, const int& y_a, const int& z_a, 
	const float* B, const int& x_b, const int& y_b, const int& z_b
);
float matricesMaxDifference
(
	const float* A, const int& x_a, const int& y_a, const int& z_a, 
	const float* B, const int& x_b, const int& y_b, const int& z_b,
	int loc[3]
);
float matrixMax(const float* A, const int& x_a, const int& y_a, const int& z_a);

float* inputMatrix(const char* filename, int* nx, int* ny, int* nz);
void copyDoubleToFloat(float* dst, const double* src, const int& s_w, const int& s_l, const int& s_h);
void copyFloatToDouble(double* dst, const float* src, const int& s_w, const int& s_l, const int& s_h);
void copyIntToFloat(float* dst, const int* src, const int& s_w, const int& s_l, const int& s_h);

#endif
