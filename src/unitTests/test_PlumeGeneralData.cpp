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
