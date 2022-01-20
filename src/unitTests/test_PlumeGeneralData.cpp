#include "test_PlumeGeneralData.h"

void test_PlumeGeneralData::setInterpMethod(std::string interpMethod,
                                            WINDSGeneralData *WGD,
                                            TURBGeneralData *TGD)
{
  std::cout << "[Plume] \t Interpolation Method set to: " << interpMethod << std::endl;
  if (interpMethod == "analyticalPowerLaw") {
    interp = new InterpPowerLaw(WGD, TGD, debug);
  } else if (interpMethod == "nearestCell") {
    interp = new InterpNearestCell(WGD, TGD, debug);
  } else if (interpMethod == "triLinear") {
    interp = new InterpTriLinear(WGD, TGD, debug);
  } else {
    std::cerr << "[ERROR] unknown interpolation method" << std::endl;
    exit(EXIT_FAILURE);
  }
  return;
}

void test_PlumeGeneralData::testInterp(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  double xPos = 10;
  double yPos = 10;
  double zPos = 1;

  double uMean = 0.0;
  double vMean = 0.0;
  double wMean = 0.0;

  double txx = 0.0;
  double txy = 0.0;
  double txz = 0.0;
  double tyy = 0.0;
  double tyz = 0.0;
  double tzz = 0.0;

  double flux_div_x = 0.0;
  double flux_div_y = 0.0;
  double flux_div_z = 0.0;

  double CoEps = 1e-6;

  // a = 2 * 2pi/Lx
  float a = 2.0 * 2.0 * M_PI / (WGD->nx * WGD->dx);
  // b = 6 * 2pi/Ly
  float b = 6.0 * 2.0 * M_PI / (WGD->ny * WGD->dy);
  // c = 4 * 2pi/Lz
  float c = 4.0 * 2.0 * M_PI / ((WGD->nz - 1) * WGD->dz);

  xPos = 10;
  yPos = 10 + .5;
  zPos = 10 + .5;
  interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);
  std::cout << cos(a * xPos) + cos(b * yPos) + sin(c * zPos) << std::endl;
  std::cout << uMean << " " << vMean << " " << wMean << std::endl;

  xPos = 10 + .5;
  yPos = 10;
  zPos = 10 + .5;
  interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);
  std::cout << cos(a * xPos) + cos(b * yPos) + sin(c * zPos) << std::endl;
  std::cout << uMean << " " << vMean << " " << wMean << std::endl;

  xPos = 10 + .5;
  yPos = 10 + .5;
  zPos = 10;
  interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);
  std::cout << cos(a * xPos) + cos(b * yPos) + sin(c * zPos) << std::endl;
  std::cout << uMean << " " << vMean << " " << wMean << std::endl;

  xPos = 10;
  yPos = 10;
  zPos = 10;
  interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);
  std::cout << cos(a * xPos) + cos(b * yPos) + sin(c * zPos) << std::endl;
  std::cout << uMean << " " << vMean << " " << wMean << std::endl;

  return;
}

void test_PlumeGeneralData::testCPU(int length)
{
  std::vector<double> A11, A12, A13, A21, A22, A23, A31, A32, A33;
  A11.resize(length, 1.0);
  A12.resize(length, 2.0);
  A13.resize(length, 3.0);
  A21.resize(length, 2.0);
  A22.resize(length, 1.0);
  A23.resize(length, 2.0);
  A31.resize(length, 3.0);
  A32.resize(length, 2.0);
  A33.resize(length, 1.0);

  auto cpuStartTime = std::chrono::high_resolution_clock::now();
  for (size_t it = 0; it < length; ++it) {
    bool tt = invert3(A11[it], A12[it], A13[it], A21[it], A22[it], A23[it], A31[it], A32[it], A33[it]);
  }
  auto cpuEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpuElapsed = cpuEndTime - cpuStartTime;
  std::cout << "\t\t CPU  elapsed time: " << cpuElapsed.count() << " s\n";

  std::cout << A11[0] << " " << A12[0] << " " << A13[0] << std::endl;
  std::cout << A21[0] << " " << A22[0] << " " << A23[0] << std::endl;
  std::cout << A31[0] << " " << A32[0] << " " << A33[0] << std::endl;

  return;
}
