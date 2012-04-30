#ifndef CUDAMATRIXIO_H
#define CUDAMATRIXIO_H

#include "matrixIO.h"

void cudaLoadDoubleToDevice(float* dst, const double* src, const int& s_w, const int& s_l, const int& s_h) {
	std::cout << dst << std::endl;
	int matrix_size = s_w * s_l * s_h * sizeof(float);
	float* temp = (float*) malloc(matrix_size);
	copyDoubleToFloat(temp, src, s_w, s_l, s_h);
	cudaMemcpy(dst, temp, matrix_size, cudaMemcpyHostToDevice);
	}

void cudaCopyFloatToDevice(float* dst, const float* src, const int& s_w, const int& s_l, const int& s_h) {
	int matrix_size = s_w * s_l * s_h * sizeof(float);

	cudaMemcpy(dst, src, matrix_size, cudaMemcpyHostToDevice);
	}

void cudaCopyDeviceToFloat(float* dst, const float* src, const int& s_w, const int& s_l, const int& s_h) {
	int matrix_size = s_w * s_l * s_h * sizeof(float);

	cudaMemcpy(dst, src, matrix_size, cudaMemcpyDeviceToHost);
	} 

void cudaOutputMatrix(const char* output_file, const float* d_ptr, const int& nx, const int& ny, const int& nz) {
	float* h_ptr = (float*) malloc(nx * ny * nz * sizeof(float));
	cudaMemcpy(h_ptr, d_ptr, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
	outputMatrix(output_file, h_ptr, nx, ny, nz);
	}

void cudaOutputVector(const char* output_file, const float* d_ptr, const int& nz) {
	float* h_ptr = (float*) malloc(nz * sizeof(float));
	cudaMemcpy(h_ptr, d_ptr, nz * sizeof(float), cudaMemcpyDeviceToHost);
	outputVector(output_file, h_ptr, nz);
	}


#endif
