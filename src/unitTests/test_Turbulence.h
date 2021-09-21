#pragma once
#include "test_WINDSGeneralData.h"
#include "test_TURBGeneralData.h"
#include "util.h"
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>


class test_Turbulence
{
public:
  std::string mainTest();

private:
  WINDSGeneralData *WGD;
  test_TURBGeneralData *TGD;

  float compError1Dx(std::vector<float> *, std::vector<float> *);
  float compError1Dy(std::vector<float> *, std::vector<float> *);
  float compError1Dz(std::vector<float> *, std::vector<float> *);
};
