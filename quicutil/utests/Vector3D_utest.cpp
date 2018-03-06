#define BOOST_TEST_MODULE argumentParsingTest
#include <boost/test/included/unit_test.hpp>

#include "util/Vector3D.h"

BOOST_AUTO_TEST_SUITE (vector3DTest) // name of the test

BOOST_AUTO_TEST_CASE (test1)
{
  sivelab::Vector3D X(1,0,0), Y(0,1,0), Z(0,0,1);

  BOOST_CHECK_CLOSE_FRACTION ( X.dot(Y), 0.0, 0.00001 ); 
  BOOST_CHECK_CLOSE_FRACTION ( X.dot(Z), 0.0, 0.00001 ); 
  BOOST_CHECK_CLOSE_FRACTION ( Y.dot(Z), 0.0, 0.00001 ); 
}

BOOST_AUTO_TEST_SUITE_END( )
