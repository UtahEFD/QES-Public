#include <math.h>
#include <stdio.h>
#include <sys/time.h>


extern "C" void cudaSum(float*,int,float*);


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
	for(int i = 0; i < v_elms; i++) {h_m[i] = 0.0;}
	float sum_to_find = 1.0;
	h_m[0] = sum_to_find;

		printf("done.\n");

		printf("Finding sum on device...");
		struct timeval start_time, end_time, diff;

	float* d_sum; cudaMalloc((void**) &d_sum, sizeof(float));
	float* d_m;   cudaMalloc((void**) &d_m, v_elms * sizeof(float));
	cudaMemcpy(d_m, h_m, v_elms * sizeof(float), cudaMemcpyHostToDevice);

		gettimeofday(&start_time, 0);

	cudaSum(d_m, v_elms, d_sum);
	cudaThreadSynchronize();

		gettimeofday(&end_time, 0);
		printf("done.\n");
		timersub(&end_time, &start_time, &diff);
		printf("cudaSum: %f sec.\n", diff.tv_sec + diff.tv_usec/1.0e6);

	float sum; cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
		printf("Device sum = %f.\n",  sum);


		printf("Finding sum on host...");
		gettimeofday(&start_time, 0);

	sum = h_m[0];
	for(int i = 1; i < v_elms; i++) {sum += h_m[i];}

		gettimeofday(&end_time, 0);
		printf("done.\n");
		timersub(&end_time, &start_time, &diff);
		printf("hostSum: %f sec.\n", diff.tv_sec + diff.tv_usec/1.0e6);
		printf("Host sum = %f.\n",  sum);

/*
	printf("\n\nTesting throughly..."); fflush(stdout);
	//Grueling stuff
	//Put a sum in every location for testing...

	int through = v_elms;
	bool* correct = new bool[v_elms];
	for(int i = 0; i < through; i++) {
		if(i > 0) {
			h_m[i-1] = h_m[i];
			h_m[i] = sum_to_find;
			}
		cudaMemcpy(d_m, h_m, v_elms * sizeof(float), cudaMemcpyHostToDevice);
		cudaSum(d_m, v_elms, d_sum);
		cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
		if(sum == sum_to_find) {correct[i] = true;}
		}
	printf("done.\n");
	printf("Tabulating results...");

	int found_correct = 0;	
	int bads_per_line = 20;
	int bads_this_line = 0;
	for(int i = 0; i < through; i++) {
		if(correct[i]) {found_correct++;}
		else {
			printf("%d, ", i);
			bads_this_line++;
			if(bads_this_line > bads_per_line) {
				bads_this_line = 0;
				printf("\n");
				}
			}
		}

	printf("...done.\n");
	printf("  found: %d", found_correct);
	printf("  trys:  %d", through);
	printf("\n");
*/
	cudaFree(d_m);
	cudaFree(d_sum);

	free(h_m);

	return 0;
	}

