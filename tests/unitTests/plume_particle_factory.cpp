//
// Created by Fabien Margairaz on 9/2/23.
//
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "plume/Particle.hpp"
#include "plume/Particle_Tracer.hpp"
#include "plume/Particle_Heavy.hpp"
#include "plume/ParticleLarge.hpp"
#include "plume/ParticleHeavyGas.hpp"
#include "plume/ParticleFactories.hpp"

TEST_CASE("particle factory", "[Working]")
{

  ParseParticle *protoParticle;
  ParticleTypeFactory *particleTypeFactory = new ParticleTypeFactory();

  std::string tracerstr = "Particle_Tracer";
  std::string smallstr = "Particle_Heavy";
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

    ParseParticle *protoParticle_Tracer = new ParseParticle_Tracer();
    for (int k = 0; k < 100000; ++k) {
      particleList[k] = particleTypeFactory->Create(protoParticle_Tracer);
      protoParticle_Tracer->setParticleParameters(particleList[k]);

      particleList[k]->m = 0.001;
      particleList[k]->m_kg = particleList[k]->m * (1.0E-3);
      particleList[k]->m_o = particleList[k]->m;
      particleList[k]->m_kg_o = particleList[k]->m * (1.0E-3);
    }
    REQUIRE(particleList[100]->d == 0.0);
    REQUIRE(particleList[500]->rho == 0.0);
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

    ParseParticle *protoParticle_Heavy = new ParseParticle_Heavy();
    protoParticle_Heavy->d = 0.0001;
    protoParticle_Heavy->rho = 0.0001;
    for (int k = 0; k < 100000; ++k) {
      particleList[k] = particleTypeFactory->Create(protoParticle_Heavy);
      protoParticle_Heavy->setParticleParameters(particleList[k]);

      particleList[k]->m = 0.001;
      particleList[k]->m_kg = particleList[k]->m * (1.0E-3);
      particleList[k]->m_o = particleList[k]->m;
      particleList[k]->m_kg_o = particleList[k]->m * (1.0E-3);
    }
    REQUIRE(particleList[100]->d == 0.0001);
    REQUIRE(particleList[500]->rho == 0.0001);
    REQUIRE(particleList[1000]->tag == smallstr);
    REQUIRE(particleList[10000]->particleType == ParticleType::heavy);

    endTime = std::chrono::high_resolution_clock::now();
    cpuElapsed = endTime - startTime;
    std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";
  }

  SECTION("Large Particles")
  {
    particleList.resize(100000, nullptr);

    startTime = std::chrono::high_resolution_clock::now();

    ParseParticle *protoParticleLarge = new ParseParticleLarge();
    protoParticleLarge->d = 0.001;
    protoParticleLarge->rho = 0.01;
    for (int k = 0; k < 100000; ++k) {
      particleList[k] = particleTypeFactory->Create(protoParticleLarge);
      protoParticleLarge->setParticleParameters(particleList[k]);
    }
    REQUIRE(particleList[100]->d == 0.001);
    REQUIRE(particleList[500]->rho == 0.01);
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
    protoParticleHeavyGas->rho = 0.1;
    for (int k = 0; k < 100000; ++k) {
      particleList[k] = particleTypeFactory->Create(protoParticleHeavyGas);
      protoParticleHeavyGas->setParticleParameters(particleList[k]);
    }
    REQUIRE(particleList[100]->d == 0.0);
    REQUIRE(particleList[500]->rho == 0.1);
    REQUIRE(particleList[1000]->tag == heavygasstr);
    REQUIRE(particleList[10000]->particleType == ParticleType::heavygas);

    endTime = std::chrono::high_resolution_clock::now();
    cpuElapsed = endTime - startTime;
    std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";
  }
}
