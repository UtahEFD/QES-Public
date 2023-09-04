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

  particleList.resize(100000, nullptr);
  ParseParticle *protoParticleTracer = new ParseParticleTracer();
  for (int k = 0; k < 100000; ++k) {
    particleList[k] = particleTypeFactory->Create(protoParticleTracer);
  }
  REQUIRE(particleList[1000]->tag == tracerstr);

  particleList.resize(100000, nullptr);
  ParseParticle *protoParticleSmall = new ParseParticleSmall();
  for (int k = 0; k < 100000; ++k) {
    particleList[k] = particleTypeFactory->Create(protoParticleSmall);
  }
  REQUIRE(particleList[1000]->tag == smallstr);

  particleList.resize(100000, nullptr);
  ParseParticle *protoParticleLarge = new ParseParticleLarge();
  for (int k = 0; k < 100000; ++k) {
    particleList[k] = particleTypeFactory->Create(protoParticleLarge);
  }
  REQUIRE(particleList[1000]->tag == largestr);

  particleList.resize(100000, nullptr);
  ParseParticle *protoParticleHeavyGas = new ParseParticleHeavyGas();
  for (int k = 0; k < 100000; ++k) {
    particleList[k] = particleTypeFactory->Create(protoParticleHeavyGas);
  }
  REQUIRE(particleList[1000]->tag == heavygasstr);

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpuElapsed = endTime - startTime;
  std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";
}