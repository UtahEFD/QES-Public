#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "util/QESout.h"


TEST_CASE("A test to show Catch2 is working for Unit Tests")
{

  QESout::splashScreen();

  int sum = 2 + 2;
  REQUIRE(sum == 4);
}

TEST_CASE("Another test to show Catch2 is working for Unit Tests")
{
  int sum = 2 + 2;
  int sub = 2 - 2;
  std::cout << "test" << std::endl;
  REQUIRE(sum == 4);
  REQUIRE(sub == 0);
}
