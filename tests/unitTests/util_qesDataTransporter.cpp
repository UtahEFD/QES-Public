#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "util/QESDataTransport.h"

TEST_CASE("Particle Data Transport Container")
{
  QESDataTransport pdc;

  // store a mass variable
  float massActualVal = 0.0213F;
  pdc.put("mass", massActualVal);

  std::vector<float> posActualVal{ 12.0f, 13.0f, 14.0f };
  float velActualVal[3] = { 0.1f, 0.2f, 0.3f };
  pdc.put("pos", posActualVal);
  pdc.put("vel", velActualVal);

  float mass = pdc.get<float>("mass");

  float eps = 1.0e-3;
  REQUIRE_THAT(mass, Catch::Matchers::WithinAbs(massActualVal, eps));

  std::vector<float> pos = pdc.get<std::vector<float>>("pos");
  REQUIRE_THAT(pos[0], Catch::Matchers::WithinAbs(posActualVal[0], eps));
  REQUIRE_THAT(pos[1], Catch::Matchers::WithinAbs(posActualVal[1], eps));
  REQUIRE_THAT(pos[2], Catch::Matchers::WithinAbs(posActualVal[2], eps));

  float *vel = pdc.get<float *>("vel");
  REQUIRE_THAT(vel[0], Catch::Matchers::WithinAbs(velActualVal[0], eps));
  REQUIRE_THAT(vel[1], Catch::Matchers::WithinAbs(velActualVal[1], eps));
  REQUIRE_THAT(vel[2], Catch::Matchers::WithinAbs(velActualVal[2], eps));

  std::vector<int> lotsOfInts;
  for (int i = 0; i < 1000; ++i) {
    lotsOfInts.push_back(i);
  }
  pdc.put("lotsOfInts", lotsOfInts);

  std::vector<int> allMyInts = pdc.get<std::vector<int>>("lotsOfInts");
  for (int idx = 0; idx < allMyInts.size(); ++idx) {
    REQUIRE(allMyInts[idx] == lotsOfInts[idx]);
  }


  // Check contains call
  REQUIRE(true == pdc.contains("mass"));
  REQUIRE(true == pdc.contains("pos"));
  REQUIRE(true == pdc.contains("vel"));
  REQUIRE(true == pdc.contains("lotsOfInts"));

  REQUIRE(false == pdc.contains("badKey1"));
  REQUIRE(false == pdc.contains("badKey2"));
  REQUIRE(false == pdc.contains("badKey3"));


  //
  // Test for keys that are not contained
  //
  bool exceptionTriggered = false;
  try {
    float f = pdc.get<float>("keyThatIsNotThere");
  } catch (const std::exception &e) {
    exceptionTriggered = true;
  }
  REQUIRE(exceptionTriggered == true);

  //
  // Test for incorrect type casts
  //
  exceptionTriggered = false;
  std::vector<int> vecOfInts(10, 1);
  pdc.put("badCastValue", vecOfInts);
  try {
    float f = pdc.get<float>("badCastValue");
  } catch (const std::exception &e) {
    exceptionTriggered = true;
  }
  REQUIRE(exceptionTriggered == true);
}
