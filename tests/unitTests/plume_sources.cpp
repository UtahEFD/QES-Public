//
// Created by Fabien Margairaz on 9/8/23.
//
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "plume/Source.hpp"
#include "plume/SourceParameters.hpp"
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

  std::vector<Source *> sources;
  for (auto s : tID->sourceParams->sources) {
    sources.push_back(new Source((int)sources.size(), s));
  }

  REQUIRE(sources[0]->sourceIdx == 0);
  REQUIRE(sources[0]->particleType() == ParticleType::large);
  REQUIRE(sources[0]->geometryType() == SourceShape::point);
  REQUIRE(sources[0]->releaseType() == ParticleReleaseType::continuous);

  REQUIRE(sources[1]->sourceIdx == 1);
  REQUIRE(sources[1]->particleType() == ParticleType::small);
  REQUIRE(sources[1]->geometryType() == SourceShape::sphereShell);
  REQUIRE(sources[1]->releaseType() == ParticleReleaseType::duration);
}
