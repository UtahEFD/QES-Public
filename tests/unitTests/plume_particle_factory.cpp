//
// Created by Fabien Margairaz on 9/2/23.
//
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "plume/Particle.hpp"
#include "plume/ParticleTracer.hpp"
#include "plume/ParticleSmall.hpp"
#include "plume/ParticleLarge.hpp"
#include "plume/ParticleHeavyGas.hpp"
#include "plume/ParticleFactories.hpp"

TEST_CASE("particle factory", "[Working]")
{

  ParseParticle *protoParticle;
  ParticleTypeFactory *particleTypeFactory = new ParticleTypeFactory();

  std::string tracerstr = "ParticleTracer";
  std::string smallstr = "ParticleSmall";
  std::string largestr = "ParticleLarge";
  std::string heavygasstr = "ParticleHeavyGas";

  auto startTime = std::chrono::high_resolution_clock::now();
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpuElapsed = endTime - startTime;

  std::vector<Particle *> particleList;

  SECTION("Tracer Particles")
  {
    particleList.resize(100000, nullptr);

    startTime = std::chrono::high_resolution_clock::now();

    ParseParticle *protoParticleTracer = new ParseParticleTracer();
    protoParticleTracer->d = 0.0001;
    protoParticleTracer->rho = 0.0001;
    protoParticleTracer->decayConst = 0.0001;
    for (int k = 0; k < 100000; ++k) {
      particleList[k] = particleTypeFactory->Create(protoParticleTracer);
      particleList[k]->d = protoParticleTracer->d;
      particleList[k]->d_m = (1.0E-6) * protoParticleTracer->d;
      particleList[k]->rho = protoParticleTracer->rho;
      particleList[k]->depFlag = protoParticleTracer->depFlag;
      particleList[k]->decayConst = protoParticleTracer->decayConst;
      particleList[k]->c1 = protoParticleTracer->c1;
      particleList[k]->c2 = protoParticleTracer->c2;

      particleList[k]->m = 0.001;
      particleList[k]->m_kg = particleList[k]->m * (1.0E-3);
      particleList[k]->m_o = particleList[k]->m;
      particleList[k]->m_kg_o = particleList[k]->m * (1.0E-3);
    }
    REQUIRE(particleList[100]->d == 0.0001);
    REQUIRE(particleList[1000]->tag == tracerstr);
    REQUIRE(particleList[10000]->particleType == ParticleType::tracer);

    endTime = std::chrono::high_resolution_clock::now();
    cpuElapsed = endTime - startTime;
    std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";
  }

  SECTION("Small Particles")
  {
    particleList.resize(100000, nullptr);

    startTime = std::chrono::high_resolution_clock::now();

    ParseParticle *protoParticleSmall = new ParseParticleSmall();
    for (int k = 0; k < 100000; ++k) {
      particleList[k] = particleTypeFactory->Create(protoParticleSmall);
      particleList[k]->d = protoParticleSmall->d;
      particleList[k]->d_m = (1.0E-6) * protoParticleSmall->d;
      particleList[k]->rho = protoParticleSmall->rho;
      particleList[k]->depFlag = protoParticleSmall->depFlag;
      particleList[k]->decayConst = protoParticleSmall->decayConst;
      particleList[k]->c1 = protoParticleSmall->c1;
      particleList[k]->c2 = protoParticleSmall->c2;

      particleList[k]->m = 0.001;
      particleList[k]->m_kg = particleList[k]->m * (1.0E-3);
      particleList[k]->m_o = particleList[k]->m;
      particleList[k]->m_kg_o = particleList[k]->m * (1.0E-3);
    }
    REQUIRE(particleList[1000]->tag == smallstr);
    REQUIRE(particleList[10000]->particleType == ParticleType::small);

    endTime = std::chrono::high_resolution_clock::now();
    cpuElapsed = endTime - startTime;
    std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";
  }

  SECTION("Large Particles")
  {
    particleList.resize(100000, nullptr);

    startTime = std::chrono::high_resolution_clock::now();

    ParseParticle *protoParticleLarge = new ParseParticleLarge();
    for (int k = 0; k < 100000; ++k) {
      particleList[k] = particleTypeFactory->Create(protoParticleLarge);
    }
    REQUIRE(particleList[1000]->tag == largestr);
    REQUIRE(particleList[10000]->particleType == ParticleType::large);

    endTime = std::chrono::high_resolution_clock::now();
    cpuElapsed = endTime - startTime;
    std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";
  }

  SECTION("Heavy Gas Particles")
  {
    particleList.resize(100000, nullptr);

    startTime = std::chrono::high_resolution_clock::now();

    ParseParticle *protoParticleHeavyGas = new ParseParticleHeavyGas();
    for (int k = 0; k < 100000; ++k) {
      particleList[k] = particleTypeFactory->Create(protoParticleHeavyGas);
    }
    REQUIRE(particleList[1000]->tag == heavygasstr);
    REQUIRE(particleList[10000]->particleType == ParticleType::heavygas);

    endTime = std::chrono::high_resolution_clock::now();
    cpuElapsed = endTime - startTime;
    std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";
  }
}
