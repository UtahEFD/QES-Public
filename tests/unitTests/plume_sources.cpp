//
// Created by Fabien Margairaz on 9/8/23.
//
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "plume/Source.hpp"
#include "plume/Sources.hpp"
#include "plume/SourceGeometry.hpp"
#include "plume/SourceGeometry_Cube.hpp"
#include "plume/SourceGeometry_FullDomain.hpp"
#include "plume/SourceGeometry_Line.hpp"
#include "plume/SourceGeometry_Point.hpp"
#include "plume/SourceGeometry_SphereShell.hpp"

#include "util/ParseInterface.h"

class testInputData : public ParseInterface
{
public:
  SourceParameters *sourceParams = nullptr;

  testInputData() = default;

  testInputData(const std::string fileName)
  {
    parseXML(fileName, "QESPlumeParameters");
  }

  virtual void parseValues()
  {
    parseElement<SourceParameters>(false, sourceParams, "sourceParameters");
  }
};

TEST_CASE("sources", "[in progress]")
{
  std::string qesPlumeParamFile = QES_DIR;
  qesPlumeParamFile.append("/tests/unitTests/plume_sources_parameters.xml");
  // ParseParticle *protoParticle;
  // ParticleTypeFactory *particleTypeFactory = new ParticleTypeFactory()

  SourceParameters *sourceParams;
  auto tID = new testInputData(qesPlumeParamFile);

  int numSources_Input = tID->sourceParams->sources.size();

  REQUIRE(numSources_Input == 2);

  auto *source1 = new Source(0, tID->sourceParams->sources.at(0));
  REQUIRE(source1->particleType() == ParticleType::large);
  REQUIRE(source1->geometryType() == SourceShape::point);
  REQUIRE(source1->releaseType() == ParticleReleaseType::continuous);

  auto *source2 = new Source(1, tID->sourceParams->sources.at(1));
  REQUIRE(source2->particleType() == ParticleType::small);
  REQUIRE(source2->geometryType() == SourceShape::sphereSurface);
  REQUIRE(source2->releaseType() == ParticleReleaseType::duration);
}
