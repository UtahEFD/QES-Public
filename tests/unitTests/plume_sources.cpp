//
// Created by Fabien Margairaz on 9/8/23.
//
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "plume/Source.hpp"
#include "plume/Source_Tracers.h"
#include "plume/Source_HeavyParticles.h"

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
  auto *particles = new ParticleContainers();

  std::cout << "Create new sources" << endl;
  for (auto s : tID->sourceParams->sources) {
    s->checkReleaseInfo(0.1, 1000);
    s->checkPosInfo(0, 100, 0, 100, 0, 100);

    switch (s->particleType()) {
    case tracer:
      sources.push_back(new Source_Tracers((int)sources.size(), s));
      break;
    case small:
      sources.push_back(new Source_HeavyParticles((int)sources.size(), s));
      break;
    case large:
      break;
    case heavygas:
      break;
    default:
      exit(1);
    }
  }

  SECTION("SOURCE 0")
  {
    REQUIRE(sources[0]->sourceIdx == 0);
    REQUIRE(sources[0]->particleType() == ParticleType::tracer);
    REQUIRE(sources[0]->geometryType() == SourceShape::point);
    REQUIRE(sources[0]->releaseType() == ParticleReleaseType::continuous);

    REQUIRE(sources[0]->getNewParticleNumber(0.1, 0) == 400);
    REQUIRE(sources[0]->getNewParticleNumber(0.1, 150) == 400);
    REQUIRE(sources[0]->getNewParticleNumber(0.1, 250) == 400);

    particles->prepare(sources[0]->particleType(), sources[0]->getNewParticleNumber(0.1, 0));
    particles->sweep();

    REQUIRE(particles->tracers->check_size(400) == true);
    REQUIRE(particles->tracers->size() == 400);
    REQUIRE(particles->get_nbr_active(sources[0]->particleType()) == 0);
    sources[0]->emitParticles(0.1, 0, particles);
    REQUIRE(particles->tracers->check_size(100) == false);
    REQUIRE(particles->tracers->size() == 400);
    REQUIRE(particles->get_nbr_active(sources[0]->particleType()) == 400);
  }

  SECTION("SOURCE 1")
  {
    REQUIRE(sources[1]->sourceIdx == 1);
    REQUIRE(sources[1]->particleType() == ParticleType::small);
    REQUIRE(sources[1]->geometryType() == SourceShape::sphereShell);
    REQUIRE(sources[1]->releaseType() == ParticleReleaseType::duration);

    REQUIRE(sources[1]->getNewParticleNumber(0.1, 0) == 0);
    REQUIRE(sources[1]->getNewParticleNumber(0.1, 150) == 400);
    REQUIRE(sources[1]->getNewParticleNumber(0.1, 250) == 0);

    particles->prepare(sources[1]->particleType(), sources[1]->getNewParticleNumber(0.1, 150));
    particles->sweep();

    REQUIRE(particles->heavy_particles->check_size(400) == true);
    REQUIRE(particles->heavy_particles->size() == 400);
    REQUIRE(particles->get_nbr_active(sources[1]->particleType()) == 0);
    sources[1]->emitParticles(0.1, 0, particles);
    REQUIRE(particles->heavy_particles->size() == 400);
    REQUIRE(particles->get_nbr_active(sources[1]->particleType()) == 0);

    REQUIRE(particles->heavy_particles->check_size(400) == true);
    sources[1]->emitParticles(0.1, 100, particles);
    REQUIRE(particles->heavy_particles->size() == 400);
    REQUIRE(particles->get_nbr_active(sources[1]->particleType()) == 400);
  }
  particles->container_info();
}
