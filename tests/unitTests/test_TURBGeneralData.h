#pragma once

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

class test_TURBGeneralData : public TURBGeneralData
{
public:
  test_TURBGeneralData(WINDSGeneralData *WGD)
    : TURBGeneralData(WGD)
  {}
  virtual ~test_TURBGeneralData()
  {}

  void test_compDerivatives_CPU(WINDSGeneralData *);
  void test_compDerivatives_GPU(WINDSGeneralData *);

private:
  test_TURBGeneralData();
};
