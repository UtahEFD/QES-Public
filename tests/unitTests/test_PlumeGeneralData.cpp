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
}
