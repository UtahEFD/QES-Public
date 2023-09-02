//
// Created by Fabien Margairaz on 9/2/23.
//
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "plume/Particle.hpp"
#include "plume/ParseParticle.hpp"
#include "plume/ParticleTracer.hpp"
#include "plume/ParticleSmall.hpp"
#include "plume/ParticleLarge.hpp"
#include "plume/ParticleHeavyGas.hpp"
#include "plume/ParticleFactories.hpp"

TEST_CASE("particle factory", "[in progress]")
{

  ParseParticle *protoParticle;
  ParticleTypeFactory *particleTypeFactory = new ParticleTypeFactory();
  ParticleTracerFactory particleTracerFactory;
  ParticleSmallFactory particleSmallFactory;
  ParticleLargeFactory particleLargeFactory;
  ParticleHeavyGasFactory particleHeavyGasFactory;

  std::string tracerstr = "ParticleTracer";
  std::string smallstr = "ParticleSmall";
  std::string largestr = "ParticleLarge";
  std::string heavygasstr = "ParticleHeavyGas";

  particleTypeFactory->RegisterParticles(tracerstr, &particleTracerFactory);
  particleTypeFactory->RegisterParticles(smallstr, &particleSmallFactory);
  particleTypeFactory->RegisterParticles(largestr, &particleLargeFactory);
  particleTypeFactory->RegisterParticles(heavygasstr, &particleHeavyGasFactory);

  auto startTime = std::chrono::high_resolution_clock::now();
  std::vector<Particle *> particleList;
  particleList.resize(100000);
  for (int k = 0; k < 100000; ++k) {
    particleList[k] = particleTypeFactory->Create(tracerstr);
  }
  for (int k = 0; k < 100000; ++k) {
    particleList[k] = particleTypeFactory->Create(smallstr);
  }
  for (int k = 0; k < 100000; ++k) {
    particleList[k] = particleTypeFactory->Create(largestr);
  }
  for (int k = 0; k < 100000; ++k) {
    particleList[k] = particleTypeFactory->Create(heavygasstr);
  }
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpuElapsed = endTime - startTime;
  std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";
}