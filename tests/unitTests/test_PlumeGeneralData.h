#pragma once

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"
#include "plume/PLUMEGeneralData.h"

#include "vectorMath.h"

class test_PlumeGeneralData : public PLUMEGeneralData
{
public:
  test_PlumeGeneralData(WINDSGeneralData *WGD, TURBGeneralData *TGD)
    : PLUMEGeneralData(WGD, TGD)
  {}
  virtual ~test_PlumeGeneralData()
  {}

  void setInterpMethod(std::string, WINDSGeneralData *, TURBGeneralData *);

  std::string testInterp(WINDSGeneralData *, TURBGeneralData *);
  std::string timeInterpCPU(WINDSGeneralData *, TURBGeneralData *);

#ifdef HAS_CUDA
  // void testGPU(int);
  // void testGPU_struct(int);
#endif

  using PLUMEGeneralData::interp;

private:
  test_PlumeGeneralData();
};
