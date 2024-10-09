#pragma once

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "test_function.h"


class test_functions
{
public:
  test_functions(WINDSGeneralData *, TURBGeneralData *, const std::string &);
  void setTestValues(WINDSGeneralData *, TURBGeneralData *);

  test_function *u_test_function;
  test_function *v_test_function;
  test_function *w_test_function;
  test_function *c_test_function;

private:
  test_functions() : domain(0, 0, 0, 0, 0, 0) {}

  qes::Domain domain;
};