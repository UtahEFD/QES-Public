#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "QESDataTransport.h"

TEST_CASE("Particle Data Transport Container")
{
  QESDataTransport pdc;

  float massActualVal = 0.0213F;
  pdc.put( "mass", massActualVal );

  std::vector<float> posActualVal{12.0f, 13.0f, 14.0f};
  float velActualVal[3] = {0.1f, 0.2f, 0.3f};
  pdc.put( "pos", posActualVal );
  pdc.put( "vel", velActualVal );

  float mass = pdc.get("mass");

  float eps = 1.0e-3;
  REQUIRE_THAT(mass, Catch::Matchers::WithinAbs(massActualVal, eps));

  std::vector<float> pos = pdc.get("pos");
  REQUIRE_THAT(pos[0], Catch::Matchers::WithinAbs(posActualVal[0], eps));
  REQUIRE_THAT(pos[1], Catch::Matchers::WithinAbs(posActualVal[1], eps));
  REQUIRE_THAT(pos[2], Catch::Matchers::WithinAbs(posActualVal[2], eps));

  float *vel = pdc.get("vel");
  REQUIRE_THAT(vel[0], Catch::Matchers::WithinAbs(velActualVal[0], eps));
  REQUIRE_THAT(vel[1], Catch::Matchers::WithinAbs(velActualVal[1], eps));
  REQUIRE_THAT(vel[2], Catch::Matchers::WithinAbs(velActualVal[2], eps));
}
