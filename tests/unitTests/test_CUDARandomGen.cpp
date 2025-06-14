#include <random>
#include <bitset>
#include <string>
#include <iostream>
#include <cstring>

#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "CUDARandomKernel.h"

// Not sure I'll include yet...
// #define CUDA_CALL(x) do { if((x)!=cudaSuccess) {              \
//            printf("Error at %s:%d\n",__FILE__,__LINE__);       \
//            return;}} while(0)

// #define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {  \
//            printf("Error at %s:%d\n",__FILE__,__LINE__);       \
//            return;}} while(0)

TEST_CASE("Create CURAND Generator")
{
  // Create pseudo-random number generator
  curandGenerator_t gen;
  curandStatus_t status;
  status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);

  REQUIRE(status == CURAND_STATUS_SUCCESS);
}


TEST_CASE("CUDA Random Gen - Avg Uniform Value")
{
  size_t i;

  curandGenerator_t gen;
  float *devPRNVals, *hostData, *devResults, *hostResults;

  // Create pseudo-random number generator
  // CURAND_CALL(
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);

  // Set the seed using random's device...
  curandSetPseudoRandomGeneratorSeed(gen, std::random_device{}());

  // start with 15k "particles"
  int numParticles = 15000;

  // repeat with ever increasing numbers of particles
  for (int r = 0; r < 15; r++) {

    // Allocate numParticle * 3 floats on host
    int n = numParticles * 3;

    hostResults = (float *)calloc(n, sizeof(float));

    // Allocate n floats on device to hold random numbers
    // CUDA_CALL();
    cudaMalloc((void **)&devPRNVals, n * sizeof(float));
    // CUDA_CALL();
    cudaMalloc((void **)&devResults, n * sizeof(float));

    // Generate n random floats on device
    // CURAND_CALL();
    // generates n vals between [0, 1]
    curandGenerateUniform(gen, devPRNVals, n);

    // uses the random numbers in a kernel and simply converts
    // them to a [-1, 1] space
    genPRNOnGPU(n, devPRNVals, devResults);

    /* Copy device memory to host */
    // CUDA_CALL(cudaMemcpy(hostData, devPRNVals, n * sizeof(float),
    // cudaMemcpyDeviceToHost));
    // CUDA_CALL();
    cudaMemcpy(hostResults, devResults, n * sizeof(float), cudaMemcpyDeviceToHost);

    // random numbers are generated between -1 and 1.  Avg should
    // be close to 0.0
    float avgVal = 0.0;
    for (i = 0; i < n; i++) {
      avgVal += hostResults[i];
    }
    avgVal /= float(n);

    float eps = 1.0e-1;
    REQUIRE_THAT(avgVal, Catch::Matchers::WithinAbs(0.0F, eps));

    numParticles *= 2;

    // CUDA_CALL();
    cudaFree(devPRNVals);

    // CUDA_CALL();
    cudaFree(devResults);

    free(hostResults);
  }

  // Cleanup
  // CURAND_CALL();
  curandDestroyGenerator(gen);
}

TEST_CASE("CUDA Random Gen - Frequency")
{
  curandGenerator_t gen;
  float *devPRNVals, *hostData, *devResults, *hostResults;

  // Create pseudo-random number generator
  // CURAND_CALL(
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);

  // Set the seed using random's device...
  curandSetPseudoRandomGeneratorSeed(gen, std::random_device{}());

  // start with 15k "particles"
  int numParticles = 15000;

  // repeat with ever increasing numbers of particles
  for (int r = 0; r < 10; r++) {

    // Allocate numParticle * 3 floats on host
    int n = numParticles * 3;

    hostResults = (float *)calloc(n, sizeof(float));

    // Allocate n floats on device to hold random numbers
    // CUDA_CALL();
    cudaMalloc((void **)&devPRNVals, n * sizeof(float));
    // CUDA_CALL();
    cudaMalloc((void **)&devResults, n * sizeof(float));

    // Generate n random floats on device
    // CURAND_CALL();
    // generates n vals between [0, 1]
    curandGenerateUniform(gen, devPRNVals, n);

    /* Copy device memory to host */
    // CUDA_CALL(cudaMemcpy(hostData, devPRNVals, n * sizeof(float),
    // cudaMemcpyDeviceToHost));
    // CUDA_CALL();
    cudaMemcpy(hostResults, devPRNVals, n * sizeof(float), cudaMemcpyDeviceToHost);

    // determine if the number of values between 0, 1 is evenly
    // distributed.

    std::vector<int> freq(10, 0);

    for (int i = 0; i < n; i++) {
      float val = hostResults[i];

      int bIdx = (int)floor(val * 10.0f);
      // inssuring that val == 1 is in the top interval (val > 1 is not possible)
      if (bIdx == 10) { bIdx = 9; }
      freq[bIdx]++;
    }

    int equalBinSize = n / 10;
    int threshold = n * 0.005;

    for (int i = 0; i < freq.size(); ++i) {
      REQUIRE_THAT(freq[i], Catch::Matchers::WithinAbs(equalBinSize, threshold));
    }

    numParticles *= 2;

    // CUDA_CALL();
    cudaFree(devPRNVals);

    // CUDA_CALL();
    cudaFree(devResults);

    free(hostResults);
  }

  // Cleanup
  curandDestroyGenerator(gen);
}


TEST_CASE("CUDA Random Gen - Frequency of Zeros and Ones")
{
  curandGenerator_t gen;
  unsigned int *devPRNVals, *hostData, *devResults, *hostResults;

  // Create pseudo-random number generator
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);

  // Set the seed using random's device...
  curandSetPseudoRandomGeneratorSeed(gen, std::random_device{}());

  // start with 15k "particles"
  int numParticles = 15000;

  // repeat with ever increasing numbers of particles
  for (int r = 0; r < 10; r++) {

    // Allocate numParticle * 3 floats on host
    int n = numParticles * 3;

    hostResults = (unsigned int *)calloc(n, sizeof(unsigned int));

    cudaMalloc((void **)&devPRNVals, n * sizeof(unsigned int));
    cudaMalloc((void **)&devResults, n * sizeof(unsigned int));

    // Generate n random integers on device - doing that in this test
    // so we can verify if random distribution of 0s and 1s is
    // approximately equal.
    //
    // If we generate floats between [0, 1], the distribution will be
    // biased due to floating point number representation.
    curandGenerate(gen, devPRNVals, n);

    // Copy device memory to host
    cudaMemcpy(hostResults, devPRNVals, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    long onesCount = 0;

    constexpr int szFloatBits = sizeof(unsigned int) * 8;

    for (int i = 0; i < n; i++) {
      unsigned int val = hostResults[i];

      uint32_t bits;
      std::memcpy(&bits, &val, sizeof(bits));

      std::bitset<szFloatBits> bitset(bits);
      std::string bitString = bitset.to_string();

      for (int c = 0; c < szFloatBits; ++c) {
        if (bitString.at(c) == '1')
          onesCount++;
      }
    }

    long numBits = n * 32;
    long zerosCount = numBits - onesCount;

    double ratio = onesCount / (double)numBits;

    // std::cout << "Ratio: " << ratio << ", ones=" << onesCount << ", zeros=" << zerosCount << std::endl;
    CHECK((0.45 < ratio && ratio < 0.55));

    numParticles *= 2;

    cudaFree(devPRNVals);
    cudaFree(devResults);

    free(hostResults);
  }

  // Cleanup
  curandDestroyGenerator(gen);
}
