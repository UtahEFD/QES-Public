#include <iostream>
#include <iomanip>
#include <sys/time.h>

texture<unsigned int, 2, cudaReadModeElementType> texture_array;

__global__ void sum(unsigned long* temp, unsigned long* result, int X, int Y) {
  
	unsigned long localSum = 0;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	for(int i = 0; i < Y; i++) {
	  localSum += tex2D(texture_array, index, i);
	}
	
	*(temp + index) = localSum;
	
	__syncthreads();
	
	localSum = 0;
	
	if(index == 0) {
	  
	  for(int i = 0; i < X; i++) {
	    localSum += *(temp + i);
	  }
	  
	  *result = localSum;
	}
  
}

int main(int argc, char** argv) {
	
	int X = 4000, Y = 4000;
	unsigned int size = X * Y * sizeof(unsigned int);
	unsigned long host_result = 0;
	unsigned int* host_array = NULL;
	unsigned long* host_temp = NULL;
	// unsigned int* device_array = NULL;
	unsigned long* device_temp = NULL;
	unsigned long* device_result = NULL;
	struct timeval start_time, end_time, diff;
	
	// Prepare the Channel Description for the texutre
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	
	// Prepare the CUDA Array to store the data.
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, X, Y);

	host_array = (unsigned int*)malloc(size);
	host_temp = (unsigned long*)malloc(X * sizeof(unsigned long));
	//if(cudaMalloc((void**)&device_array, size) != cudaSuccess) {
	//  std::cout << "Could not allocate matrix on device!" << std::endl;
	//}
	if(cudaMalloc((void**)&device_temp, X * sizeof(unsigned long)) != cudaSuccess) {
	  std::cout << "Could not allocate total array on device!" << std::endl;
	}
	if(cudaMalloc((void**)&device_result, sizeof(unsigned long)) != cudaSuccess) {
	  std::cout << "Could not allocate result on device!" << std::endl;
	}
	
	std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(8);
	
	std::cout << "Populating host matricies..." << std::flush;
	for(int y = 0; y < Y; y++) {
		for(int x = 0; x < X; x++) {
			*(host_array + y * Y + x) = (y * Y + x);
		}
	}
	for(int x = 0; x < X; x++) {
		*(host_temp + x) = 0;
	}
	std::cout << " done." << std::endl;
	
	std::cout << "Copying host matrix to device..." << std::flush;
	//if(cudaMemcpy(device_array, host_array, size, cudaMemcpyHostToDevice) != cudaSuccess) {
	//  std::cout << " Could not transfer matrix to device!" << std::flush;
	//}
	if(cudaMemcpyToArray(cuArray, 0, 0, host_array, size, cudaMemcpyHostToDevice) != cudaSuccess) {
	  std::cout << " Could not transfer matrix to device texture!" << std::flush;
	}
	if(cudaMemcpy(device_temp, host_temp, X * sizeof(unsigned long), cudaMemcpyHostToDevice) != cudaSuccess) {
	  std::cout << " Could not transfer total array to device!" << std::endl;
	}
	if(cudaMemcpy(device_result, &host_result, sizeof(unsigned long), cudaMemcpyHostToDevice) != cudaSuccess) {
	  std::cout << " Could not transfer total array to device!" << std::endl;
	}
	std::cout << " done." << std::endl;
	
	// Set texture parameters
	texture_array.addressMode[0] = cudaAddressModeClamp;
	texture_array.addressMode[1] = cudaAddressModeClamp;
	texture_array.filterMode     = cudaFilterModePoint;
	texture_array.normalized     = false;

	// Bind the array to the array
	cudaBindTextureToArray(texture_array, cuArray, channelDesc);

	/*
	std::cout << "Displaying matrix..." << std::endl;
	for(int y = 0; y < Y; y++) {
		for(int x = 0; x < X; x++) {
			std::cout << *(host_array + y * Y + x) << " ";
		}
		std::cout << std::endl;
	}
	*/
	
	std::cout << "Calculating sum on host..." << std::flush;
	gettimeofday(&start_time, 0);
	for(int y = 0; y < Y; y++) {
		for(int x = 0; x < X; x++) {
			host_result += *(host_array + y * Y + x);
		}
	}
	gettimeofday(&end_time, 0); 
	std::cout << " done." << std::endl;
	timersub(&end_time, &start_time, &diff); 
	std::cout << "    Time: " << diff.tv_sec + diff.tv_usec/1.0e6 << " sec" << std::endl;
	
	std::cout << "Running CUDA kernel..." << std::flush;
	gettimeofday(&start_time, 0);
	sum<<<8, 500>>>(device_temp, device_result, X, Y);
	cudaThreadSynchronize();
	gettimeofday(&end_time, 0);
	cudaError_t errNo = cudaGetLastError();
	if(errNo != cudaSuccess) {
	  std::cout << " KERNAL LAUNCH FAILED! (" << errNo << ")" << std::endl;
	} else {
	  std::cout << " done." << std::endl;
	}
	timersub(&end_time, &start_time, &diff); 
	std::cout << "    Time: " << diff.tv_sec + diff.tv_usec/1.0e6 << " sec" << std::endl;
	
	
	std::cout << std::setprecision(1) << "Host total: " << host_result << std::endl;
	
	unsigned long device_result_on_host;
	cudaMemcpy(&device_result_on_host, device_result, sizeof(unsigned long), cudaMemcpyDeviceToHost);
	std::cout << "Device total: " << device_result_on_host << std::endl;
	
	if(device_result_on_host == host_result) {
	  std::cout << "OK!" << std::endl;
	} else {
	  std::cout << "NO MATCH!" << std::endl;
	}
	
	free(host_array);
	free(host_temp);
	cudaFree(device_temp);
	cudaFree(device_result);
	cudaFreeArray(cuArray);
		
	return 0;
}

