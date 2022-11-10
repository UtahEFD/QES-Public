#pragma once

#include "util.h"
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"
#include "plume/Plume.hpp"
#include "util.h"

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

  void setTestFunctions(WINDSGeneralData *, TURBGeneralData *);

  float (test_PlumeGeneralData::*u_testFunction)(WINDSGeneralData *, float, float, float);
  float (test_PlumeGeneralData::*v_testFunction)(WINDSGeneralData *, float, float, float);
  float (test_PlumeGeneralData::*w_testFunction)(WINDSGeneralData *, float, float, float);
  float (test_PlumeGeneralData::*c_testFunction)(WINDSGeneralData *, float, float, float);

  float testFunction_linearX(WINDSGeneralData *, float, float, float);
  float testFunction_linearY(WINDSGeneralData *, float, float, float);
  float testFunction_linearZ(WINDSGeneralData *, float, float, float);
  float testFunction_trig(WINDSGeneralData *, float, float, float);

  std::string testInterp(WINDSGeneralData *, TURBGeneralData *);
  std::string timeInterpCPU(WINDSGeneralData *, TURBGeneralData *);
  //void test_compDerivatives_CPU(WINDSGeneralData *);
  //void test_compDerivatives_GPU(WINDSGeneralData *);
  void testGPU(int);
  void testGPU_struct(int);

  void testCPU(int);

private:
  test_PlumeGeneralData();
};
