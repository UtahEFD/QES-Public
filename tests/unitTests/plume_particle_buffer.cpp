//
// Created by Fabien Margairaz on 9/21/23.
//
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <list>
#include <chrono>

#include "plume/ManagedContainer.h"
#include "plume/ParticleIDGen.h"
// #include "plume/ParticleManager.h"

#include "plume/Particle.h"
#include "plume/TracerParticle.h"

#include "plume/Random.h"

void scrubParticleList(std::list<Particle *> &particleList)
{
  for (auto parItr = particleList.begin(); parItr != particleList.end();) {
    if ((*parItr)->state != ACTIVE) {
      delete *parItr;
      parItr = particleList.erase(parItr);
    } else {
      ++parItr;
    }
  }
}

void advect(Particle *p)
{
  p->velMean = { 10, 0, 0 };
}

TEST_CASE("buffer", "[in progress]")
{

  std::list<Particle *> particleList;

  auto start = std::chrono::high_resolution_clock::now();

  for (int k = 0; k < 10000; ++k) {
    std::list<Particle *> nextSetOfParticles;
    for (int pidx = 0; pidx < 1000; ++pidx) {

      Particle *cPar = new TracerParticle();
      cPar->state = ACTIVE;
      nextSetOfParticles.push_front(cPar);
    }
    particleList.insert(particleList.end(), nextSetOfParticles.begin(), nextSetOfParticles.end());

    Random prng;
    for (auto p : particleList) {
      float t = prng.uniRan();
      advect(p);
      if (t > 0.8)
        p->state = INACTIVE;
    }
    scrubParticleList(particleList);
  }

  std::cout << particleList.size() << std::endl;

  auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  std::chrono::duration<float> elapsed = finish - start;
  std::cout << "elapsed time: " << elapsed.count() << " s\n";

  /* FM - OBSOLETE
  start = std::chrono::high_resolution_clock::now();
  ParticleBuffer particleBuffer(1000);

  for (int k = 0; k < 10000; ++k) {
    particleBuffer.check_size(1000);

    for (int pidx = 0; pidx < 1000; ++pidx) {
      // size_t tmp = particleBuffer.next();
      // particleBuffer.buffer[tmp] = Particle_Tracer();
      // particleBuffer.buffer[tmp].isActive = true;
      particleBuffer.add();
    }

    for (auto &p : particleBuffer.buffer) {
      float t = drand48();
      advect(&p);
      if (t > 0.8)
        p.isActive = false;
    }
    particleBuffer.scrub_buffer();
  }

  std::cout << particleBuffer.size() << " " << particleBuffer.nbr_used << std::endl;

  finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  elapsed = finish - start;
  std::cout << "elapsed time: " << elapsed.count() << " s\n";

  REQUIRE(particleBuffer.size() == 6000);
  */

  start = std::chrono::high_resolution_clock::now();
  ManagedContainer<TracerParticle> tracers;
  ParticleIDGen *id_gen = ParticleIDGen::getInstance();
  int new_tracers = 1E3;
  for (int k = 0; k < 10000; ++k) {
    tracers.check_resize(new_tracers);

    for (int pidx = 0; pidx < new_tracers; ++pidx) {
      tracers.insert();
      tracers.last_added()->ID = id_gen->get();
    }
    Random prng;
    for (auto &tracer : tracers) {
      float t = prng.uniRan();
      advect(&tracer);
      if (t > 0.8)
        tracer.state = INACTIVE;
    }
  }

  std::cout << tracers.size() << " " << tracers.get_nbr_active() << std::endl;

  finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  elapsed = finish - start;
  std::cout << "elapsed time: " << elapsed.count() << " s\n";

  REQUIRE(tracers.last_added()->ID == 10000 * new_tracers - 1);
  REQUIRE(tracers.size() == 6000);
}

TEST_CASE("buffer large", "[in progress]")
{
  auto start = std::chrono::high_resolution_clock::now();

  ManagedContainer<TracerParticle> tracers(1E5);
  std::vector<float> tmp(1E5);
  for (int k = 0; k < 1E4; ++k) {
    tracers.check_resize(2E3);
    tracers.resize_companion(tmp);

    /*for (int pidx = 0; pidx < 2E3; ++pidx) {
      tracers.insert();
    }*/

    std::vector<size_t> newParticleIndices;
    tracers.obtain_available(2E3, newParticleIndices);
    for (size_t n = 0; k < newParticleIndices.size(); ++k) {
      tracers[newParticleIndices[n]].state = ACTIVE;
    }

    Random prng;
    for (auto &tracer : tracers) {
      float t = prng.uniRan();
      advect(&tracer);
      if (t > 0.8)
        tracer.state = INACTIVE;
    }
  }

  std::cout << tracers.size() << " " << tracers.get_nbr_active() << std::endl;
  std::cout << tracers.size() << " " << tmp.size() << std::endl;

  auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  std::chrono::duration<float> elapsed = finish - start;
  std::cout << "elapsed time: " << elapsed.count() << " s\n";
}
