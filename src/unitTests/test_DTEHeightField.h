#pragma once
#include "DTEHeightField.h"
#include "util.h"
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>


class test_DTEHeightField
{
public:
  std::string mainTest();
  std::string testCutCells();

private:
  DTEHeightField DTEHF;
};
