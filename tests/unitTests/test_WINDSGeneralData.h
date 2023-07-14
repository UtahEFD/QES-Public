#pragma once

#include "winds/WINDSGeneralData.h"
#include "util.h"
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>


class test_WINDSGeneralData : public WINDSGeneralData
{
public:
  test_WINDSGeneralData()
  {}
  test_WINDSGeneralData(const int[3], const float[3]);
  test_WINDSGeneralData(const int[3], const float[3], const float *);
  virtual ~test_WINDSGeneralData()
  {}

private:
};
