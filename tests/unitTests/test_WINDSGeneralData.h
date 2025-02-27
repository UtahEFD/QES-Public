#pragma once

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "winds/WINDSGeneralData.h"

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
