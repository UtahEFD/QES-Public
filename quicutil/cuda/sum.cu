/* File: sum.cu
 * Author: Joshua Clark
 * Date: 11/10/2009
 * 
 * The following is example code to sum a continues chunk of memory (treated as 
 * 2D matrix). This code is the base for an example of using texture memory 
 * within CUDA. Please refer to texture.cu to see the modifications necessary
 * to use texture memory.
 *
 */

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <limits>

/* This is the sum kernel that runs on the GPU. This kernel is extremely 
 * inefficient and uses a temporary memory space to store the results from the
 * initial round of summing.
 *
 * input is the 2D matrix that needs to be summed, this is a pointer to the 
 * spot in global (device) memory where the matrix is located.
 *
 * temp is a temporary array (which has been initialized to zero) that stores
 * the results from the initial round of summation.
 *
 * result is the final value that is read back into main memory as the result
 * of the summation.
 *
 * X & Y are the dimensions of the 2D matrix.
 *
 */
__global__ void sum(unsigned int* input, unsigned long* temp, 
                    unsigned long* result, int X, int Y) {
  
  // localSum is a temporary variable to use while summing.
	unsigned long localSum = 0;
	
	/* index is the unique identifier for a thread, it is important to note that
	 * threadIdx.x only refers to the index within a block.
	 */
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Loop through and sum an entire column.
	for(int i = 0; i < Y; i++) {
	  localSum += *(input + index * X + i);
	}
	
	// Set the result within the temp array.
	*(temp + index) = localSum;
	
	// Make sure all the threads are finished before you start summing temp.
	__syncthreads();
	
	// Now, have one thread sum the partial sums contained within temp and then
	// store the final result.
	if(index == 0) {
	  localSum = 0;
	  
	  for(int i = 0; i < X; i++) {
	    localSum += *(temp + i);
	  }
	  
	  *result = localSum;
	}
  
}

int main(int argc, char** argv) {
	
  std::cout << "max unsigned int: " << std::numeric_limits<unsigned int>::max() << std::endl;

	int X = 4000, Y = 4000;
	unsigned int size = X * Y * sizeof(unsigned int);
	unsigned long host_result = 0;
	unsigned int* host_array = NULL;
	unsigned int* host_temp = NULL;
	unsigned int* device_array = NULL;
	unsigned long* device_temp = NULL;
	unsigned long* device_result = NULL;
	struct timeval start_time, end_time, diff;
	
	host_array = (unsigned int*)malloc(size);
	host_temp = (unsigned int*)malloc(X * sizeof(unsigned int));
	if(cudaMalloc((void**)&device_array, size) != cudaSuccess) {
	  std::cout << "Could not allocate matrix on device!" << std::endl;
	}
	if(cudaMalloc((void**)&device_temp, X * sizeof(unsigned long)) != cudaSuccess) {
	  std::cout << "Could not allocate total array on device!" << std::endl;
	}
	if(cudaMalloc((void**)&device_result, sizeof(unsigned long)) != cudaSuccess) {
	  std::cout << "Could not allocate result on device!" << std::endl;
	}
	
	// Creating texutre...
	// texture<float, 1, cudaReadModeElementType> device_tex;
	
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
	if(cudaMemcpy(device_array, host_array, size, cudaMemcpyHostToDevice) != cudaSuccess) {
	  std::cout << " Could not transfer matrix to device!" << std::flush;
	}
	if(cudaMemcpy(device_temp, host_temp, X * sizeof(unsigned long), cudaMemcpyHostToDevice) != cudaSuccess) {
	  std::cout << " Could not transfer total array to device!" << std::endl;
	}
	if(cudaMemcpy(device_result, &host_result, sizeof(unsigned long), cudaMemcpyHostToDevice) != cudaSuccess) {
	  std::cout << " Could not transfer total array to device!" << std::endl;
	}
	std::cout << " done." << std::endl;
	
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
	sum<<<8, 500>>>(device_array, device_temp, device_result, X, Y);
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
	cudaFree(device_array);
	cudaFree(device_temp);
	cudaFree(device_result);
	
	return 0;
}

