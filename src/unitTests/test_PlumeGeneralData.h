#pragma once

#include "util.h"
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"
#include "plume/Plume.hpp"

#include "vectorMath.h"


class test_PlumeGeneralData : public Plume
{
public:
  test_PlumeGeneralData(WINDSGeneralData *WGD, TURBGeneralData *TGD)
    : Plume(WGD, TGD)
  {}
  virtual ~test_PlumeGeneralData()
  {}

  void setInterpMethod(std::string, WINDSGeneralData *, TURBGeneralData *);
  void testInterp(WINDSGeneralData *, TURBGeneralData *);
  //void test_compDerivatives_CPU(WINDSGeneralData *);
  //void test_compDerivatives_GPU(WINDSGeneralData *);
  void testGPU(int);
  void testGPU_struct(int);

  void testCPU(int);

private:
  test_PlumeGeneralData();
};
