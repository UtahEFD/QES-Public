#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Verify GPU Accessibility")
{
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  REQUIRE( error_id == cudaSuccess );
  
  REQUIRE( deviceCount >= 1 );

  int dev, driverVersion = 0, runtimeVersion = 0;
  for (dev = 0; dev < deviceCount; ++dev) {

    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << "\tDevice " << dev << ": " << deviceProp.name << std::endl;

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "\t\tCUDA Driver Version / Runtime Version: "
              << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << " / "
              << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;

    std::cout << "\t\tCUDA Capability Major/Minor version number: "
              << deviceProp.major << "." << deviceProp.minor << std::endl;

    char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(msg, sizeof(msg),
              "\t\tTotal amount of global memory: %.0f MBytes "
              "(%llu bytes)\n",
              static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
              (unsigned long long)deviceProp.totalGlobalMem);
#else
    snprintf(msg, sizeof(msg),
             "\t\tTotal amount of global memory: %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
#endif
    std::cout << msg;

    //    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
    //           deviceProp.multiProcessorCount,
    //           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
    //           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
    //           deviceProp.multiProcessorCount);

    std::cout << "\t\tGPU Max Clock rate:  "
              << deviceProp.clockRate * 1e-3f << " MHz ("
              << deviceProp.clockRate * 1e-6f << " GHz)" << std::endl;

    std::cout << "\t\tPCI: BusID=" << deviceProp.pciBusID << ", "
              << "DeviceID=" << deviceProp.pciDeviceID << ", "
              << "DomainID=" << deviceProp.pciDomainID << std::endl;
  }
  cudaSetDevice(0);


  std::cout << "Version: " << QES_VERSION_INFO << std::endl;// QES_VERSION_INFO comes from CMakeLists.txt
#ifdef HAS_CUDA
  std::cout << "\t* CUDA support available!" << std::endl;
#else
  std::cout << "* No CUDA support - CPU Only Computations!" << std::endl;
#endif

#ifdef HAS_OPTIX
  std::cout << "\t* OptiX is available!" << std::endl;
#endif


}

TEST_CASE("Test VectorMath GPU")
{
  
}
