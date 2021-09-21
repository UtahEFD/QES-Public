#pragma once

#include "WINDSGeneralData.h"
#include "TURBGeneralData.h"
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

  void test_compDerivatives(WINDSGeneralData *);

private:
  test_TURBGeneralData();
};
