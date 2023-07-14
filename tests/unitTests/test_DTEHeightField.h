#pragma once

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "winds/DTEHeightField.h"

#define TEST_PASS ""

class test_DTEHeightField
{
public:
  std::string mainTest();
  std::string testCutCells();

private:
  DTEHeightField DTEHF;
};
