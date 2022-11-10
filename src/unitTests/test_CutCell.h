#pragma once
#include "winds/Cut_cell.h"
#include "util.h"
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>


class test_CutCell
{
public:
  std::string mainTest();
  std::string testCalculateAreaTopBot();

private:
  Cut_cell cutCell;
};
