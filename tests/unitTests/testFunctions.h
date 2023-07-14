#pragma once

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "testFunction.h"


class testFunctions
{
public:
  testFunctions(WINDSGeneralData *, TURBGeneralData *, const std::string &);
  void setTestValues(WINDSGeneralData *, TURBGeneralData *);

  testFunction *u_testFunction;
  testFunction *v_testFunction;
  testFunction *w_testFunction;
  testFunction *c_testFunction;

private:
  testFunctions() {}
};