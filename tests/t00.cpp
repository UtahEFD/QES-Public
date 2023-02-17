#include <catch2/catch_test_macros.hpp>

TEST_CASE( "A test to show Catch2 is working for Unit Tests" )
{
  int sum = 2 + 2;
  REQUIRE( sum == 4 );
}
