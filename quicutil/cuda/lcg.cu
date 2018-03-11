#include <iostream>
#include <iomanip>
#include <limits>
#include <cassert>
#include <vector>
#include <sys/time.h>

#include "math_constants.h"

// float computeMean(const std::vector<int> &v);
float computeMean(float *arr, int N);

// float computeStd(const std::vector<int> &v);
float computeStd(float *arr, int N);

__device__ inline float randomLCG(unsigned int *lcg_x)
{
  // from Num. Recipes in C: lcgCoef_a=1664525, lcgCoef_c=1013904223
  unsigned int lcgCoef_a = 1664525;	
  unsigned int lcgCoef_c = 1013904223;
  unsigned int lcgCoef_m = 4294967295; // from C++ --> std::numeric_limits<unsigned int>::max();

  *lcg_x = (lcgCoef_a * *lcg_x + lcgCoef_c) % lcgCoef_m;

  return (float)(*lcg_x / (float)lcgCoef_m);
}


__device__ inline float2 boxMuller(float rval1, float rval2)
{
  float r = sqrt(-2.f*log(rval1));  
  float theta=2.0f * CUDART_PI_F * rval2;  
  return make_float2(r*sin(theta),r*cos(theta));  
}

__global__ void lcgFill(unsigned int* lcgCoef_x, float* randArray)
{
  /* index is the unique identifier for a thread, it is important to note that
   * threadIdx.x only refers to the index within a block.
   */
  int index = threadIdx.x; // blockIdx.x * blockDim.x + threadIdx.x;
  
  // use the index to lookup lcgCoef_x
  unsigned int lcg_x = lcgCoef_x[index];

  float2 nrVal = boxMuller( randomLCG(&lcg_x), randomLCG(&lcg_x) ); 
  randArray[index] = nrVal.x;

  lcgCoef_x[index] = lcg_x;
}

int main(int argc, char** argv) 
{
  srand48( time(0) % getpid() );

  int X = 10, Y = 10;

  // For LCG, each thread will need a single coefficient "x" to
  // maintain a separate generator for PRNs.
  // These values will be seeded initially on the host, and then copied
  // to the device for use by the threads in computing PRNs.

  // Ideally, each thread needs it's own coefficient.
  unsigned long N = X * Y;
  unsigned long numBytes = N * sizeof(unsigned int);

  std::vector< std::vector<int> > bins(N);
  for (unsigned int idx=0; idx<bins.size(); idx++)
    {
      bins[idx].resize(10);
      std::fill(bins[idx].begin(), bins[idx].end(), 0);  
    }

  unsigned int* host_array = 0;
  float* rand_array_on_host = 0;

  unsigned int* device_array = 0;
  float* rand_array = 0;

  std::cout << "Creating " << N << " unsigned ints... num bytes=" << numBytes / 1024 << "KiB" << std::endl;

  // allocate storage on the host
  host_array = new unsigned int[N];
  rand_array_on_host = new float[N];
  std::cout << "rand_array_on_host = " << rand_array_on_host << std::endl;

  // allocate storage on the device/gpu
  if(cudaMalloc((void**)&device_array, numBytes) != cudaSuccess) {
    std::cout << "Could not allocate matrix on device!" << std::endl;
  }

  // allocate storage on the device/gpu for the PRNs
  if(cudaMalloc((void**)&rand_array, N * sizeof(float)) != cudaSuccess) {
    std::cout << "Could not allocate matrix on device!" << std::endl;
  }

  std::cout << "Populating initial seeds for PRNG..." << std::flush;
  for(unsigned long idx=0; idx<N; idx++) 
    {
      host_array[idx] = (unsigned int)floor(drand48() * std::numeric_limits<unsigned int>::max());
    }
  std::cout << " done." << std::endl;
	
  std::cout << "Copying host matrix to device..." << std::flush;
  if(cudaMemcpy(device_array, host_array, numBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cout << " Could not transfer matrix to device!" << std::flush;
  }
  std::cout << " done." << std::endl;
  delete [] host_array;
  
  int runCount = 0;
  while (runCount < 1000)
    {
      // std::cout << "Running CUDA kernel..." << std::flush;
      lcgFill<<<1, N>>>(device_array, rand_array);
      cudaThreadSynchronize();

      cudaError_t errNo = cudaGetLastError();
      if(errNo != cudaSuccess) {
	std::cout << " KERNAL LAUNCH FAILED! (\"" << cudaGetErrorString(errNo) << "\")" << std::endl;
      } else {
	// std::cout << " done." << std::endl;
      }
      
      if (cudaMemcpy(rand_array_on_host, rand_array, N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
	  std::cout << " Could not transfer rand array from device!" << std::flush;
	}

      // analyze the random sequence
      // for (int idx=0; idx<N; idx++)
      // {
	  // std::cout << "rval = " << rand_array_on_host[idx] << std::endl;
	  // int binIdx = (int)floor(rand_array_on_host[idx] * 10);
	  // bins[idx][binIdx]++;
      // }

      std::cout << "Run " << runCount << ": Mean=" << computeMean(rand_array_on_host, N) 
		<< ", Std=" << computeStd(rand_array_on_host, N) << std::endl;

      runCount++;
    }

#if 0
  for (unsigned int bIdx=0; bIdx<bins.size(); bIdx++)
    {
      std::cout << "Bin " << bIdx << ": Mean=" << computeMean(bins[bIdx]) << ", Std=" << computeStd(bins[bIdx]) << std::endl;
      std::fill(bins[bIdx].begin(), bins[bIdx].end(), 0);  
    }
#endif
	
  cudaFree(device_array);
  cudaFree(rand_array);

  return 0;
}



float computeMean(float *arr, int N)
{
  int sum=0;
  for (unsigned int i=0; i<N; i++)
    sum += arr[i];
  return sum / (float)N;
}

float computeStd(float *arr, int N)
{
  assert(N > 1);

  float mean = computeMean(arr, N);

  float tmp=0.0;
  for (unsigned int i=0; i<N; i++)
    tmp += ((arr[i] - mean) * (arr[i] - mean));

  return sqrt( tmp / (N - 1) );
}
