/* File: sum.cu
 * Author: Joshua Clark
 * Date: 11/10/2009
 * 
 * The following is example code to sum a continuous chunk of memory (treated as 
 * 2D matrix). This code is the base for an example of using texture memory 
 * within CUDA. Please refer to texture.cu to see the modifications necessary
 * to use texture memory.
 *
 */

#include <iostream>
#include <iomanip>
#include <limits>
#include <sys/time.h>

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
__global__ void sum(unsigned long* tauSeeds, unsigned long* temp, 
                    unsigned long* result, int X, int Y) 
{
  /* index is the unique identifier for a thread, it is important to note that
   * threadIdx.x only refers to the index within a block.
   */
  int index = blockIdx.x * blockDim.x + threadIdx.x;
	
  tauSeeds + index;

  // Loop through and sum an entire column.
  for(int i = 0; i < Y; i++) {
    localSum += *(input + index * X + i);
  }
	
  // Set the result within the temp array.
  *(temp + index) = localSum;

  // generate random number
  float rVal =   
      unsigned long tVal = 2.3283064365387e-10                      // Periods
	* (tausStep(m_taus_z1, 13, 19, 12, 4294967294UL) ^   // p1 = 2^31-1
	   tausStep(m_taus_z2, 2, 25, 4, 4294967288UL) ^     // p2 = 2^30-1
	   tausStep(m_taus_z3, 3, 11, 17, 4294967280UL) ^    // p3 = 2^28-1
	   lcgStep(m_taus_z4, 1664525, 1013904223UL)         // p4 = 2^32
	   );
      return (float)((tVal) / (double)std::numeric_limits<unsigned int>::max());
    }

combinedTausworthe();
  
}

int main(int argc, char** argv) 
{
  int X = 100, Y = 100;

  // Each thread will need 4 longs to hold the z1, z2, z3, and z4
  // state values of the combinedTausworthe PRNG.  We will seed these
  // values initially on the host, and then copy them to the device
  // for use by the threads in computing PRNs.
  unsigned int N = X * Y * 4;
  unsigned int numBytes = N * sizeof(unsigned long);
  unsigned long* host_array = 0;
  unsigned long* device_array = 0;

  std::cout << "Creating 4 * " << X * Y << " unsigned longs... num bytes=" << numBytes / 1024 << "KiB" << std::endl;

  // allocate storage on the host
  host_array = new unsigned long[N];

  // allocate storage on the device/gpu
  if(cudaMalloc((void**)&device_array, numBytes) != cudaSuccess) {
    std::cout << "Could not allocate matrix on device!" << std::endl;
  }

  std::cout << "Populating initial seeds for PRNG..." << std::flush;
  for(unsigned int idx=0; idx<N; idx++) 
    {
      host_array[idx] = (unsigned long)(drand48() * std::numeric_limits<unsigned long>::max());
    }
  std::cout << " done." << std::endl;
	
  std::cout << "Copying host matrix to device..." << std::flush;
  if(cudaMemcpy(device_array, host_array, numBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cout << " Could not transfer matrix to device!" << std::flush;
  }
  std::cout << " done." << std::endl;
	
    std::cout << "Displaying matrix..." << std::endl;
    for(unsigned int idx=0; idx<N; idx++) {
      std::cout << "host_array[" << idx << "] = " << host_array[idx] << std::endl;
    }

  delete [] host_array;
  
#if 0
  std::cout << "Running CUDA kernel..." << std::flush;
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
#endif
	
  cudaFree(device_array);

  return 0;
}

