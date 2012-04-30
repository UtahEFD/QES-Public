#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>

#include "../../Tools/matrixIO.h"

//#include "../Kernels/abs_diff.cu"
extern "C" void cudaAbsDiff(float*,float*,float*,int);
extern "C" void cudaFindSum(float*,int,float*);


int main(int argc, char* argv[]) {

//Make a test volume...(for the device)

		std::cout << "Making a \"volume\"..." << std::flush;

	int v_wide = 128;
	int v_long = 201;
	int v_tall = 21;

	int v_elms = v_wide * v_long * v_tall;

	float* h_m = (float*) malloc(v_elms * sizeof(float));
	for(int i = 0; i < v_elms; i++) {h_m[i] = 0.0;}

	float* h_n = (float*) malloc(v_elms * sizeof(float));
	for(int i = 0; i < v_elms; i++) {h_n[i] = 1.0;}

		std::cout << "done." << std::endl;

		std::cout << "Finding absolute difference..." << std::flush;

		struct timeval start_time, end_time, diff;

	float* d_m; cudaMalloc((void**) &d_m, v_elms * sizeof(float));
	cudaMemcpy(d_m, h_m, v_elms * sizeof(float), cudaMemcpyHostToDevice);

	float* d_n; cudaMalloc((void**) &d_n, v_elms * sizeof(float));
	cudaMemcpy(d_n, h_n, v_elms * sizeof(float), cudaMemcpyHostToDevice);

	float* d_absdif; cudaMalloc((void**) &d_absdif, v_elms * sizeof(float));
	cudaMemcpy(d_absdif, d_m, v_elms * sizeof(float), cudaMemcpyDeviceToDevice);

	float* d_tabsdif; cudaMalloc((void**) &d_tabsdif, sizeof(float));

		gettimeofday(&start_time, 0);

	cudaAbsDiff(d_absdif, d_m, d_n, v_elms);

		gettimeofday(&end_time, 0);

		std::cout << "done." << std::endl;

		timersub(&end_time, &start_time, &diff);
		std::cout << "cudaAbsDiff: " << diff.tv_sec + diff.tv_usec/1.0e6 << " sec" << std::endl;

	cudaFindSum(d_absdif, v_elms, d_tabsdif);
	float tabsdif; cudaMemcpy(&tabsdif, d_tabsdif, sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Total absolute difference = " << tabsdif << "." << std::endl;

	std::cout << std::endl << std::endl;

	std::cout << "Testing throughly..." << std::flush;

	//Grueling stuff
	//Put a sum in every location for testing...

	int through = v_elms;
	float ttl_abs_dif_to_find = 1.0;
	bool* correct = new bool[v_elms];
	h_m[0] = ttl_abs_dif_to_find;
	for(int i = 0; i < through; i++) {
		if(i > 0) {
			h_m[i-1] = h_m[i];
			h_m[i] = ttl_abs_dif_to_find;
			}
		//cudaMemcpy(d_m, h_m, v_elms * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_n, h_m, v_elms * sizeof(float), cudaMemcpyHostToDevice);
		cudaAbsDiff(d_absdif, d_m, d_n, v_elms);
		cudaFindSum(d_absdif, v_elms, d_tabsdif);
		cudaMemcpy(&tabsdif, d_tabsdif, sizeof(float), cudaMemcpyDeviceToHost);
		if(tabsdif == ttl_abs_dif_to_find) {correct[i] = true;}
		}
	std::cout << "done." << std::endl;
	std::cout << "Tabulating results: " << std::endl;

	int found_correct = 0;	
	int bads_per_line = 20;
	int bads_this_line = 0;
	for(int i = 0; i < through; i++) {
		if(correct[i]) {found_correct++;}
		else {
			std::cout << i << ", " << std::flush;
			bads_this_line++;
			if(bads_this_line > bads_per_line) {
				bads_this_line = 0;
				std::cout << std::endl;
				}
			}
		}

	std::cout << std::endl;
	std::cout << "...done." << std::endl;
	std::cout << "  right: " << found_correct << std::endl;
	std::cout << "  trys:  " << through << std::endl;
	std::cout << std::endl;
	
	cudaFree(d_m);
	cudaFree(d_n);
	cudaFree(d_absdif);
	cudaFree(d_tabsdif);
	free(h_m);	
	free(h_n);
	}

