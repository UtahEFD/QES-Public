#pragma once

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"
#include "util.h"
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>


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
