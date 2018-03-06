#define BOOST_TEST_MODULE argumentParsingTest
#include <boost/test/included/unit_test.hpp>
#include "util/ArgumentParsing.h"

BOOST_AUTO_TEST_SUITE (argumentParsingTest) // name of the test

BOOST_AUTO_TEST_CASE (test1)
{
  sivelab::ArgumentParsing arg;
  BOOST_CHECK(0 == 0);
}

BOOST_AUTO_TEST_SUITE_END( )
