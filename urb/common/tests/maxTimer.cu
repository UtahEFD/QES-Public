#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>

#include "../../Tools/matrixIO.h"
//#include "../Kernels/max.cu"
extern "C" void cudaFindMax(float*,int,float*);
void hostFindMax(float*,int,float*);

int main(int argc, char* argv[]) {

		std::cout << "Making a \"volume\"..." << std::flush;
	int v_wide = 128;
	int v_long = 128;
	int v_tall = 64;

	int v_elms = v_wide * v_long * v_tall;

	float* h_m = (float*) malloc(v_elms * sizeof(float));
	for(int i = 0; i < v_elms; i++) {h_m[i] = 0.2;}
	h_m[0] = 7.0;	
		std::cout << "done." << std::endl;

	struct timeval start_time, end_time, diff;


	// With host
		std::cout << "Finding max with host..." << std::flush;
	float max_host = 17.17;

		gettimeofday(&start_time, 0);
	hostFindMax(h_m, v_elms, &max_host);
		gettimeofday(&end_time, 0);

		std::cout << "done." << std::endl;

		timersub(&end_time, &start_time, &diff);
		std::cout << "hostFindMax: " << diff.tv_sec + diff.tv_usec/1.0e6 << " sec" << std::endl;
		std::cout << "Max = " << max_host << "." << std::endl;
		std::cout << std::endl << std::endl;



	//With device
		std::cout << "Finding max with device..." << std::flush;
	float max_device = 17.17;
	float* d_max; cudaMalloc((void**) &d_max, sizeof(float));

	float* d_m; cudaMalloc((void**) &d_m, v_elms * sizeof(float));
	cudaMemcpy(d_m, h_m, v_elms * sizeof(float), cudaMemcpyHostToDevice);

		gettimeofday(&start_time, 0);
	cudaFindMax(d_m, v_elms, d_max);
		gettimeofday(&end_time, 0);

		std::cout << "done." << std::endl;

	cudaMemcpy(&max_device, d_max, sizeof(float), cudaMemcpyDeviceToHost);

		timersub(&end_time, &start_time, &diff);
		std::cout << "cudaFindMax: " << diff.tv_sec + diff.tv_usec/1.0e6 << " sec" << std::endl;
		std::cout << "Max = " << max_device << "." << std::endl;
		std::cout << std::endl << std::endl;

	free(h_m);	
	}

void hostFindMax(float* h_array, int size, float* h_max) {
	float max = h_array[0];
	for(int i = 1; i < size; i++) {if(h_array[i] > max) {max = h_array[i];}}
	*h_max = max;
	}

