#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>

#include "../../Tools/matrixIO.h"
//#include "../Kernels/sum.cu"

extern "C" void cudaFindSum(float*,int,float*);
void hostFindSum(float*,int,float*);

int main(int argc, char* argv[]) {

		std::cout << "Making a \"volume\"..." << std::flush;
	int v_wide = 257;
	int v_long = 257;
	int v_tall = 65;
	float elm_val = 1.0;

	int v_elms = v_wide * v_long * v_tall;

	float* h_m = (float*) malloc(v_elms * sizeof(float));
	for(int i = 0; i < v_elms; i++) {h_m[i] = elm_val;}

		std::cout << "done." << std::endl;

	struct timeval start_time, end_time, diff;


	// With host
		std::cout << "Finding sum with host..." << std::flush;
	float sum_host = 0.0;
	
		gettimeofday(&start_time, 0);
	hostFindSum(h_m, v_elms, &sum_host); //Doesn't do .1 well.
		gettimeofday(&end_time, 0);

		std::cout << "done." << std::endl;

		timersub(&end_time, &start_time, &diff);
		std::cout << "hostFindSum: " << diff.tv_sec + diff.tv_usec/1.0e6 << " sec" << std::endl;
		std::cout << "Sum = " << sum_host << "." << std::endl;
		std::cout << "*Doesn't do 0.1 well.*" << std::endl;
		std::cout << std::endl << std::endl;



	//With device
		std::cout << "Finding sum with device..." << std::flush;
	float sum_device = 0.0;
	float* d_sum; cudaMalloc((void**) &d_sum, sizeof(float));

	float* d_m; cudaMalloc((void**) &d_m, v_elms * sizeof(float));
	cudaMemcpy(d_m, h_m, v_elms * sizeof(float), cudaMemcpyHostToDevice);

		gettimeofday(&start_time, 0);
	cudaFindSum(d_m, v_elms, d_sum);
		gettimeofday(&end_time, 0);

		std::cout << "done." << std::endl;

	cudaMemcpy(&sum_device, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

		timersub(&end_time, &start_time, &diff);
		std::cout << "cudaFindSum: " << diff.tv_sec + diff.tv_usec/1.0e6 << " sec" << std::endl;
		std::cout << "Sum = " << (int) sum_device << "." << std::endl;
		std::cout << "Correct? : " << std::flush;
		if(v_elms * elm_val == sum_device) {std::cout << "Yes" << std::endl;}
		else							   {std::cout << "No" << std::endl;}
		std::cout << std::endl << std::endl;

	free(h_m);	
	}

void hostFindSum(float* h_array, int size, float* h_sum) {
	*h_sum = 0.0;
	for(int i = 0; i < size; i++) {*h_sum += h_array[i];}
	}

