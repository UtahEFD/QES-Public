#pragma once

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "winds/Cut_cell.h"


class test_CutCell
{
public:
  std::string mainTest();
  std::string testCalculateAreaTopBot();

private:
  Cut_cell cutCell;
};
