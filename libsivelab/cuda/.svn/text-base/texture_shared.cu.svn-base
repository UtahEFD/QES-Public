#include <iostream>
#include <iomanip>
#include <sys/time.h>

// This is the reference to the "texture" on the GPU, it should be noted that
// the texture simply referrs to a location in memory that already exists and
// has data in it (like a cuda array for example). This reference must be
// declared global and is accessed by both the kerenel and the method 
// responcible for setting up the texture.
texture<unsigned int, 2, cudaReadModeElementType> texture_array;

// This kernel will first have each thread sum a chunk of data, then one thread
// in each block will sum those partial sums, then store that result back out
// to global memory.
__global__ void partialSum(unsigned long* gtemp, int chunkSize, int threadCount) {
  
  __shared__ unsigned long stemp[512];
	
	// localSum is a temporary variable used in order to sum the chunk this
	// particular thread is responcible for.
	unsigned long localSum = 0;
	
	// This is the index or unique identifier of this particular thread, note that
	// this is a one dimensional index.
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Actaully transfer the value from the texture memory and add it to our 
	// local running sum.
	for(int i = 0; i < chunkSize; i++) {
	  localSum += tex2D(texture_array, index, i);
	}
	
	// Write back the results from the sum of this chunk back to shared memory,
	// this is happening across a block at a time.
	stemp[threadIdx.x] = localSum;
	
	// Synchronize all of the threads, so that one thread can then sum the partial
	// sums within this block.
	__syncthreads();
	
	// Thread zero (in this block, you can only synchronize across blocks!) now
	// sums the partial sums for this block and then writes the result back to
	// global memory.
	if(threadIdx.x == 0) {
	  localSum = 0;
	  
	  for(int i = 0; i < threadCount; i++) {
	    localSum += stemp[i];
	  }
	  
	  *(gtemp + blockIdx.x) = localSum;
	  
	}
	
}

// This kernel will sum the partial sums from the partialSum kernel launch. This
// kernel is designed to simply run in serial because their are a trivial amount
// of partial sums. Because the data is still on the GPU, it is sitll faster to
// have the GPU run in serial rather than transfer all the sums back and have
// the CPU sum them.
__global__ void sumPartials(unsigned long* partials, unsigned long* result, int size) {
  
  unsigned long localSum = 0;
  
  for(int i = 0; i < size; i++) {
	    localSum += partials[i];
	}
	
	*result = localSum;
  
}

int main(int argc, char** argv) {
	
	std::cout << "\n";
	
	// X & Y are the size of the array, theoretically you just need one "size"
	// and you could sum any linear space in memory.
	int X = 4000, Y = 4000;
	unsigned int size = X * Y * sizeof(unsigned int);
	
	// This is the number of threads per block, note that this can not exceed 512
	// because of the use of shared memory within the partialSum kernel.
	int threadCount = 500;
	
	// This number can be varied in order to improve performance, I found that 
	// blockCount = 16 and chunkSize = 2000 seemed to work best but with a change
	// in data size this might need to be re-adjusted.
	int blockCount = 16; // 100 (for cudaProf)
	int chunkSize = 2000; // 320 (for cudaProf)
	
	unsigned long host_result = 0;
	unsigned int* host_array = NULL;
	unsigned long* host_temp = NULL;
	unsigned long* device_result = NULL;
	unsigned long* device_temp = NULL;
	
	// This is what stores the matrix on the device.
	cudaArray* cuArray;
	
	// The following varaibles are used for timing.
	struct timeval start_time, end_time, diff;
	
	// Prepare the Channel Description for the texutre.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	
	// Prepare the CUDA Array to store the data.
	cudaMallocArray(&cuArray, &channelDesc, threadCount * blockCount, chunkSize);
	host_array = (unsigned int*)malloc(size);
	host_temp = (unsigned long*)malloc(blockCount * sizeof(unsigned long));
	if(cudaMalloc((void**)&device_temp, blockCount * sizeof(unsigned long)) != cudaSuccess) {
	  std::cout << "Could not allocate result on device!" << std::endl;
	}
	if(cudaMalloc((void**)&device_result, sizeof(unsigned long)) != cudaSuccess) {
	  std::cout << "Could not allocate result on device!" << std::endl;
	}
	
	// Initialize host data.
	for(int y = 0; y < Y; y++) {
		for(int x = 0; x < X; x++) {
			*(host_array + y * Y + x) = (y * Y + x);
		}
	}
	for(int i = 0; i < blockCount; i++) {
	  *(host_temp + i) = 0;
	}
	
	// Copy initialized data to the device, this needs to be done for the results
	// and temporary pointers because those need to be set to zero.
	if(cudaMemcpyToArray(cuArray, 0, 0, host_array, size, cudaMemcpyHostToDevice) != cudaSuccess) {
	  std::cout << "Could not transfer matrix to device texture!" << std::endl;
	}
	if(cudaMemcpy(device_temp, &host_temp, blockCount * sizeof(unsigned long), cudaMemcpyHostToDevice) != cudaSuccess) {
	  std::cout << "Could not transfer total array to device!" << std::endl;
	}
	if(cudaMemcpy(device_result, &host_result, sizeof(unsigned long), cudaMemcpyHostToDevice) != cudaSuccess) {
	  std::cout << "Could not transfer total array to device!" << std::endl;
	}
	
	
	// Set texture parameters
	texture_array.addressMode[0] = cudaAddressModeClamp;
	texture_array.addressMode[1] = cudaAddressModeClamp;
	texture_array.filterMode     = cudaFilterModePoint;
	texture_array.normalized     = false;

	// Bind the array to the array
	cudaBindTextureToArray(texture_array, cuArray, channelDesc);
  
  // Calculate the sum on the host in a linear fashion.
  std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(8);
	gettimeofday(&start_time, 0);
	for(int y = 0; y < Y; y++) {
		for(int x = 0; x < X; x++) {
			host_result += *(host_array + y * Y + x);
		}
	}
	gettimeofday(&end_time, 0); 
	timersub(&end_time, &start_time, &diff);
	std::cout << "Host Time: " << diff.tv_sec + diff.tv_usec/1.0e6 << " sec." << std::endl;
	
	// Calculate teh sum on the device, note that we are doing a double kernel
	// launch and that we are synchronizing (which is essential for timing).
	gettimeofday(&start_time, 0);
	partialSum<<<blockCount, threadCount>>>(device_temp, chunkSize, threadCount);
	sumPartials<<<1, 1>>>(device_temp, device_result, blockCount);
	cudaThreadSynchronize();
	gettimeofday(&end_time, 0);
	cudaError_t errNo = cudaGetLastError();
	if(errNo != cudaSuccess) {
	  std::cout << " KERNAL LAUNCH FAILED! (" << errNo << ")" << std::endl;
	}
	timersub(&end_time, &start_time, &diff); 
	std::cout << "GPU Time:  " << diff.tv_sec + diff.tv_usec/1.0e6 << " sec.\n" << std::endl;
  
  /*
  // Display the partial sums (for debugging).
  cudaMemcpy(host_temp, device_temp, blockCount * sizeof(unsigned long), cudaMemcpyDeviceToHost);
  for(int i = 0; i < blockCount; i++) {
	  std::cout << *(host_temp + i) << std::endl;
	}
	*/
  
  // Display the results and get the total from the device.
	std::cout << std::setprecision(1) << "Host total:   " << host_result << std::endl;
	unsigned long device_result_on_host;
	cudaMemcpy(&device_result_on_host, device_result, sizeof(unsigned long), cudaMemcpyDeviceToHost);
	std::cout << "Device total: " << device_result_on_host << std::endl;
	
	std::cout << "\n";
	
	// Check to make sure our results are the same.
	if(device_result_on_host == host_result) {
	  std::cout << "(OK!)\n" << std::endl;
	} else {
	  std::cout << "(NOT A MATCH!)\n" << std::endl;
	}
	
	// Preform cleanup.
	free(host_array);
	cudaFree(device_result);
	cudaFreeArray(cuArray);
		
	return 0;
}

sult_on_host;
	cudaMemcpy(&device_result_on_host, device_result, sizeof(unsigned long), cudaMemcpyDeviceToHost);
	std::cout << "Device total: " <