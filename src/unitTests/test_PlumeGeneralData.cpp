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

  mat3 tmp = { 1, 2, 3, 2, 1, 2, 3, 2, 1 };
  std::vector<mat3> A;
  A.resize(length, tmp);

  std::vector<vec3> b;
  b.resize(length, { 1.0, 1.0, 1.0 });

  std::vector<vec3> x;
  x.resize(length, { 0.0, 0.0, 0.0 });

  std::vector<mat3sym> tau;
  //tau.resize(length, { 1, 2, 3, 1, 2, 1 });
  tau.resize(length, { 1, 0, 3, 0, 0, 1 });
  std::vector<vec3> invar;
  invar.resize(length, { 0.0, 0.0, 0.0 });

  auto cpuStartTime = std::chrono::high_resolution_clock::now();
  for (size_t it = 0; it < length; ++it) {
    bool tt = vectorMath::invert3(A[it]);
    vectorMath::matmult(A[it], b[it], x[it]);
  }

  for (size_t it = 0; it < length; ++it) {
    vectorMath::makeRealizable(10e-4, tau[it]);
    vectorMath::calcInvariants(tau[it], invar[it]);
  }

  auto cpuEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpuElapsed = cpuEndTime - cpuStartTime;
  std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";

  std::cout << A[0]._11 << " " << A[0]._12 << " " << A[0]._13 << std::endl;
  std::cout << A[0]._21 << " " << A[0]._22 << " " << A[0]._23 << std::endl;
  std::cout << A[0]._31 << " " << A[0]._32 << " " << A[0]._33 << std::endl;
  std::cout << x[0]._1 << " " << x[0]._2 << " " << x[0]._3 << std::endl;

  std::cout << std::endl;

  std::cout << tau[0]._11 << " " << tau[0]._12 << " " << tau[0]._13 << std::endl;
  std::cout << tau[0]._12 << " " << tau[0]._22 << " " << tau[0]._23 << std::endl;
  std::cout << tau[0]._13 << " " << tau[0]._23 << " " << tau[0]._33 << std::endl;
  std::cout << invar[0]._1 << " " << invar[0]._2 << " " << invar[0]._3 << std::endl;

  return;
}
