#include <string>
#include <cstdio>
#include <cstdlib>

#include "util.h"
#include "test_DTEHeightField.h"

int main()
{
  std::string results;
  bool testsPassed = true;

  /******************
   * DTEHeightField *
   ******************/
  printf("======================================\n");
  printf("starting DTEHeightField tests...\n");
  test_DTEHeightField DTEHF_test;
  results = DTEHF_test.mainTest();
  if (results == "") {
    printf("DTEHeightField: Success!\n");
  } else {
    printf("DTEHeightField: Failure\n%s\n", results.c_str());
    if (testsPassed) testsPassed = false;
  }

  printf("======================================\n");
  if (testsPassed) {
    printf("All tests pass!\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("At least one test failed!\n");
    exit(EXIT_FAILURE);
  }

  return 0;
}
