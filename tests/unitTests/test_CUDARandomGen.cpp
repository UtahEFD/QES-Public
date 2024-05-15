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
    status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    REQUIRE(status == CURAND_STATUS_SUCCESS);
}


TEST_CASE("CUDA Random Gen - Avg Uniform Value")
{
    size_t i;

    curandGenerator_t gen;
    float *devPRNVals, *hostData, *devResults, *hostResults;
    
    // Create pseudo-random number generator
    // CURAND_CALL(
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    
    // Set the seed --- not sure how we'll do this yet in general
    // CURAND_CALL(
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // start with 15k "particles"
    int numParticles = 15000;
    
    // repeat with ever increasing numbers of particles
    for (int r=0; r<15; r++) {

        // Allocate numParticle * 3 floats on host
        int n = numParticles * 3;

        hostResults = (float *)calloc(n, sizeof(float));
    
        // Allocate n floats on device to hold random numbers
	// CUDA_CALL();
	cudaMalloc((void **)&devPRNVals, n*sizeof(float));
	// CUDA_CALL();
	cudaMalloc((void **)&devResults, n*sizeof(float));

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
        for(i = 0; i < n; i++) {
            avgVal += hostResults[i];
        }
        avgVal /= float(n);

	float eps = 1.0e-1;
        REQUIRE_THAT( avgVal, Catch::Matchers::WithinAbs(0.0F, eps) );

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
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    
    // Set the seed --- not sure how we'll do this yet in general
    // CURAND_CALL(
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // start with 15k "particles"
    int numParticles = 15000;
    
    // repeat with ever increasing numbers of particles
    for (int r=0; r<10; r++) {

        // Allocate numParticle * 3 floats on host
        int n = numParticles * 3;

        hostResults = (float *)calloc(n, sizeof(float));
    
        // Allocate n floats on device to hold random numbers
	// CUDA_CALL();
	cudaMalloc((void **)&devPRNVals, n*sizeof(float));
	// CUDA_CALL();
	cudaMalloc((void **)&devResults, n*sizeof(float));

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

        std::vector< int > freq(10, 0);
        
        for(int i=0; i < n; i++) {
            float val = hostResults[i];

            int bIdx = (int)floor(val * 10);
            freq[ bIdx ]++;
        }

        int equalBinSize = n / 10;
        int threshold = n * 0.002;
        
        for (int i=0; i < freq.size(); ++i) {
            REQUIRE_THAT( freq[i], Catch::Matchers::WithinAbs(equalBinSize, threshold) );
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

