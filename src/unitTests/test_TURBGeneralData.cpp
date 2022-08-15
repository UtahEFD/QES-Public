#include "test_TURBGeneralData.h"

void test_TURBGeneralData::test_compDerivatives_CPU(WINDSGeneralData *WGD)
{
  auto cpuStartTime = std::chrono::high_resolution_clock::now();
  derivativeVelocity();
  auto cpuEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpuElapsed = cpuEndTime - cpuStartTime;
  std::cout << "\t\t CPU Derivatives: elapsed time: " << cpuElapsed.count() << " s\n";
}

void test_TURBGeneralData::test_compDerivatives_GPU(WINDSGeneralData *WGD)
{
  auto gpuStartTime = std::chrono::high_resolution_clock::now();
  getDerivativesGPU();
  auto gpuEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
  std::cout << "\t\t GPU Derivatives: elapsed time: " << gpuElapsed.count() << " s\n";
}
