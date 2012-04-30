#include <math.h>
#include <stdio.h>
#include <sys/time.h>


extern "C" void cudaMax(float*,int,float*);


int main(int argc, char* argv[]) 
{

		printf("Making a \"volume\"...");

	int v_wide = 256;
	int v_long = 256;
	int v_tall = 256;

	int v_elms = v_wide * v_long * v_tall;

	float* h_m = (float*) malloc(v_elms * sizeof(float));
	for(int i = 0; i < v_elms; i++) {h_m[i] = i * 0.1;}

		printf("done.\n");

		printf("Finding max on device...");

	float* d_max; cudaMalloc((void**) &d_max, sizeof(float));
	float* d_m; cudaMalloc((void**) &d_m, v_elms * sizeof(float));
	cudaMemcpy(d_m, h_m, v_elms * sizeof(float), cudaMemcpyHostToDevice);

		struct timeval start_time, end_time, diff;
		gettimeofday(&start_time, 0);

	cudaMax(d_m, v_elms, d_max);

		gettimeofday(&end_time, 0);
		printf("done.\n");
		timersub(&end_time, &start_time, &diff);
		printf("cudaMax: %f sec.\n", diff.tv_sec + diff.tv_usec/1.0e6);

	float max; cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device Max = %f.\n", max);

		printf("Finding max on host...");
		gettimeofday(&start_time, 0);

	max = h_m[0];
	for(int i = 1; i < v_elms; i++)
	{
		if(h_m[i] > max) max = h_m[i];
	}

		gettimeofday(&end_time, 0);
		printf("done.\n");
		timersub(&end_time, &start_time, &diff);
		printf("hostMax: %f sec.\n", diff.tv_sec + diff.tv_usec/1.0e6);
		printf("Host Max = %f.\n", max);


	printf("\n\n");

	printf("Testing cudaMax throughly..."); fflush(stdout);

	//Grueling stuff
	//Put a max in every location for testing...

	int through = v_elms;
	float max_to_find = 7.;
	bool* correct = new bool[v_elms];
	h_m[0] = max_to_find;
	for(int i = 0; i < through; i++) {
		if(i > 0) {
			h_m[i-1] = h_m[i];
			h_m[i] = max_to_find;
			}
		cudaMemcpy(d_m, h_m, v_elms * sizeof(float), cudaMemcpyHostToDevice);
		cudaMax(d_m, v_elms, d_max);
		cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
		if(max == max_to_find) {correct[i] = true;}
		}
	printf("done.\n");
	printf("Tabulating results..."); fflush(stdout);

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
	printf("  found: %d\n", found_correct);
	printf("  trys:  %d\n", through);
	printf("\n");	

	cudaFree(d_m);
	cudaFree(d_max);
	free(h_m);	
}

