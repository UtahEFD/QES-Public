#include <catch2/catch_test_macros.hpp>
#include "CUDARandomKernel.h"

// #include <iostream>

// #define CUDA_CALL(x) do { if((x)!=cudaSuccess) {              \
//            printf("Error at %s:%d\n",__FILE__,__LINE__);       \
//            return;}} while(0)
	    
// #define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {  \
//            printf("Error at %s:%d\n",__FILE__,__LINE__);       \
//            return;}} while(0)

TEST_CASE("Create CURAND Generator")
{
  /* Create pseudo-random number generator */
  curandGenerator_t gen;
  curandStatus_t status;
  status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  REQUIRE(status == CURAND_STATUS_SUCCESS);
}


TEST_CASE("CUDA Random Gen")
{
    size_t i;

    curandGenerator_t gen;
    float *devPRNVals, *hostData, *devResults, *hostResults;
    
    /* Create pseudo-random number generator */
    // CURAND_CALL(
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    
    /* Set seed */
    // CURAND_CALL(
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // start with 15k par
    int numParticles = 15000;
    
    for (int r=0; r<15; r++) {

        /* Allocate numParticle * 3 floats on host */
        int n = numParticles * 3;

        hostResults = (float *)calloc(n, sizeof(float));
    
        /* Allocate n floats on device */
	// CUDA_CALL();
	cudaMalloc((void **)&devPRNVals, n*sizeof(float));
	// CUDA_CALL();
	cudaMalloc((void **)&devResults, n*sizeof(float));

        /* Generate n random floats on device */
        // CURAND_CALL();
	curandGenerateUniform(gen, devPRNVals, n);

        genPRNOnGPU(n, devPRNVals, devResults);

        /* Copy device memory to host */
        // CUDA_CALL(cudaMemcpy(hostData, devPRNVals, n * sizeof(float),
        // cudaMemcpyDeviceToHost));
        // CUDA_CALL();
        cudaMemcpy(hostResults, devResults, n * sizeof(float), cudaMemcpyDeviceToHost);
    
        /* Show result */
        float avgVal = 0.0;
        for(i = 0; i < n; i++) {
            // std::cout << hostResults[i] << std::endl;
            avgVal += hostResults[i];
        }
        avgVal /= float(n);

	float eps = 1.0e-1;
	REQUIRE( avgVal < eps );
        REQUIRE( avgVal > -eps );

        numParticles *= 2;

        // CUDA_CALL();
        cudaFree(devPRNVals);
        
        // CUDA_CALL();
        cudaFree(devResults);
        
        free(hostResults);
    }

    /* Cleanup */
    // CURAND_CALL();
    curandDestroyGenerator(gen);
}
