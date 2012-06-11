#include <math.h>
#include <stdio.h>
#include <sys/time.h>

//#include "../../../../Tools/matrixIO.h"

extern "C" void cudaPack(float*, float*,float*,float*,float*,int);
extern "C" void cudaZero(float*, int, float);


int main(int argc, char* argv[]) 
{

//Make a test volumes...(for the device)

		printf("Making \"volumes\"...");

	int v_wide = 64;
	int v_long = 64;
	int v_tall = 21;

	int v_elms = v_wide * v_long * v_tall;

	float* h_1 = (float*) malloc(v_elms * sizeof(float));
	float* h_2 = (float*) malloc(v_elms * sizeof(float));
	float* h_3 = (float*) malloc(v_elms * sizeof(float));
	float* h_4 = (float*) malloc(v_elms * sizeof(float));

	float* h_packee = (float*) malloc(v_elms * 4 * sizeof(float));

	for(int i = 0; i < v_elms; i++) 
	{
		h_1[i] = 1.0;
		h_2[i] = 2.0;
		h_3[i] = 3.0;
		h_4[i] = 4.0;
		h_packee[i] = 0.0;
	}

		printf("done.\n");

		printf("Packing...");

		struct timeval start_time, end_time, diff;

	float* d_1; cudaMalloc((void**) &d_1, v_elms * sizeof(float));
	float* d_2; cudaMalloc((void**) &d_2, v_elms * sizeof(float));
	float* d_3; cudaMalloc((void**) &d_3, v_elms * sizeof(float));
	float* d_4; cudaMalloc((void**) &d_4, v_elms * sizeof(float));

	float* d_packee; cudaMalloc((void**) &d_packee, v_elms * 4 * sizeof(float));
	cudaZero(d_packee, v_elms * 4, 0.);

	cudaMemcpy(d_1, h_1, v_elms * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_2, h_2, v_elms * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_3, h_3, v_elms * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_4, h_4, v_elms * sizeof(float), cudaMemcpyHostToDevice);

		gettimeofday(&start_time, 0);

	cudaPack(d_packee, d_1, d_2, d_3, d_4, v_elms);

		gettimeofday(&end_time, 0);

		printf("done.\n");

		timersub(&end_time, &start_time, &diff);
		printf("cudaPack: %f sec.\n", (diff.tv_sec + diff.tv_usec/1.0e6));

	cudaMemcpy(h_packee, d_packee, v_elms * 4 * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < 16; i++)
	{
		printf("%f\t", h_packee[i]);
		if(i % 4 == 3) {printf("\n");}
	}

	cudaFree(d_1);
	cudaFree(d_2);
	cudaFree(d_3);
	cudaFree(d_4);
	cudaFree(d_packee);

	free(h_1);
	free(h_2);
	free(h_3);
	free(h_4);
	free(h_packee);

	printf("\n");
	return 0;
}

