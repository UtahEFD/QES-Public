#include <math.h>
#include <stdio.h>
#include <sys/time.h>

//#include "../../../../Tools/matrixIO.h"

extern "C" void cudaZero(float*, size_t, float);

int main(int argc, char* argv[]) 
{

//Make a test volume...(for the device)

	unsigned int v_wide = 256;
	unsigned int v_long = 256;
	unsigned int v_tall = 256;

		if(argc == 4)
		{
			v_wide = atoi(argv[1]);
			v_long = atoi(argv[2]);
			v_tall = atoi(argv[3]);
		}

		printf("Making a \"volume\" of size %dx%dx%d...", v_wide, v_long, v_tall);


	int v_elms = v_wide * v_long * v_tall;

	float* h_m = (float*) malloc(v_elms * sizeof(float));
	for(int i = 0; i < v_elms; i++) {h_m[i] = 0.2;}

		printf("done.\n");

		printf("Zeroing on device...");

	float* d_m; cudaMalloc((void**) &d_m, v_elms * sizeof(float));
	cudaMemcpy(d_m, h_m, v_elms * sizeof(float), cudaMemcpyHostToDevice);

		struct timeval start_time, end_time, diff;
		gettimeofday(&start_time, 0);

	cudaZero(d_m, v_elms, 0.);

		gettimeofday(&end_time, 0);
		printf("done.\n");

		timersub(&end_time, &start_time, &diff);
		printf("cudaZero: %f sec\n", diff.tv_sec + diff.tv_usec/1.0e6);


		printf("Zeroing on host...");
	gettimeofday(&start_time, 0);
	for(int i = 0; i < v_elms; i++)	{h_m[i] = 0.0f;}
	gettimeofday(&end_time, 0);
		printf("done.\n");

		timersub(&end_time, &start_time, &diff);
		printf("cudaZero: %f sec\n", diff.tv_sec + diff.tv_usec/1.0e6);;

		printf("Checking device zeroing...");
	cudaMemcpy(h_m, d_m, v_elms * sizeof(float), cudaMemcpyDeviceToHost);
	bool zeroed = true;
	for(int i = 0; i < v_elms; i++)
	{
		if(h_m[i] != 0.0)
		{
			zeroed = false;
			printf("h_m[%d] = %0.1f.\n", i, h_m[i]);
			break;
		}
	}

	(zeroed) ? printf("Passed.\n"): printf("Failed.\n");
	
	cudaFree(d_m);
	free(h_m);	
}

