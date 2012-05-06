#include <stdio.h>

void listDeviceCount() {
	int count = -1;
	cudaGetDeviceCount(&count);
	printf("Total # of devices: %d", count);
	}

void listDeviceInfo(int d) {
	if(d < 0) {printf("Device numbers start at 0. %d is not a valid device number.", d);}

	int count = -1;
	cudaGetDeviceCount(&count);
	
	if(count <= d) {printf("There are only %d devices. %d is not a valid device number.", count, d);}

	struct cudaDeviceProp d_info;
	cudaGetDeviceProperties(&d_info, d);

	printf("\n***************** Device %d *****************\n", d);
	printf("%s\n", d_info.name);
	printf("\tGlobMem: %ld\n", d_info.totalGlobalMem);
	printf("\tSharedMemPerBlock: %ld\n", d_info.sharedMemPerBlock);
	printf("\tRegistersPerBlock: %d\n", d_info.regsPerBlock);
	printf("\tWarp Size: %d\n", d_info.warpSize);
	printf("\tMemory Pitch: %ld\n", d_info.memPitch);
	printf("\tmaxThreadsPerBlock: %d\n", d_info.maxThreadsPerBlock);
	printf
	(
		"\tmaxThreadsDim: %d x %d x %d\n",
		d_info.maxThreadsDim[0],
		d_info.maxThreadsDim[1],
		d_info.maxThreadsDim[2]
	);
	printf
	(
		"\tmaxGridSize: %d x %d x %d\n", 
		d_info.maxGridSize[0],
		d_info.maxGridSize[1],
		d_info.maxGridSize[2]
	);
	printf("\ttotalConstMem: %ld\n", d_info.totalConstMem);
	printf("\tMajor: %d\n", d_info.major);
	printf("\tMinor: %d\n", d_info.minor);
	printf("\tclockRate: %d\n", d_info.clockRate);
	printf("\ttextureAlignment: %ld\n", d_info.textureAlignment);
	printf("\tdeviceOverlap: %d\n", d_info.deviceOverlap);
	printf("\tmultiProcessorCount: %d\n", d_info.multiProcessorCount);
	printf("********************************************\n");
	}

int main(int argc, char *argv[]) {
	printf("Get and List some Device info using Cuda.\n");
	
	listDeviceCount();
	
	int c = -1;
	cudaGetDeviceCount(&c);
	for(int i = 0; i < c; i++) {
		listDeviceInfo(i);
		printf("\n\n");
		}
	}

