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

#include "util/ManagedContainer.h"
// #include "plume/ParticleManager.h"

#include "plume/Particle.h"
#include "plume/TracerParticle.h"

void scrubParticleList(std::list<Particle *> &particleList)
{
  for (auto parItr = particleList.begin(); parItr != particleList.end();) {
    if (!(*parItr)->isActive) {
      delete *parItr;
      parItr = particleList.erase(parItr);
    } else {
      ++parItr;
    }
  }
}

void advect(Particle *p)
{
  p->uMean = 10;
}

TEST_CASE("buffer", "[in progress]")
{

  std::list<Particle *> particleList;

  auto start = std::chrono::high_resolution_clock::now();

  for (int k = 0; k < 10000; ++k) {
    std::list<Particle *> nextSetOfParticles;
    for (int pidx = 0; pidx < 1000; ++pidx) {

      Particle *cPar = new TracerParticle();
      cPar->isActive = true;
      nextSetOfParticles.push_front(cPar);
    }
    particleList.insert(particleList.end(), nextSetOfParticles.begin(), nextSetOfParticles.end());

    for (auto p : particleList) {
      float t = drand48();
      advect(p);
      if (t > 0.8)
        p->isActive = false;
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

  for (int k = 0; k < 10000; ++k) {
    tracers.sweep(1000);

    for (int pidx = 0; pidx < 1000; ++pidx) {
      tracers.insert();
    }
    for (auto &tracer : tracers) {
      float t = drand48();
      advect(&tracer);
      if (t > 0.8)
        tracer.isActive = false;
    }
  }

  std::cout << tracers.size() << " " << tracers.get_nbr_active() << std::endl;

  finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  elapsed = finish - start;
  std::cout << "elapsed time: " << elapsed.count() << " s\n";

  REQUIRE(tracers.last_added()->particleID == 10000 * 1000 - 1);
  REQUIRE(tracers.size() == 6000);
}

TEST_CASE("buffer large", "[in progress]")
{
  auto start = std::chrono::high_resolution_clock::now();

  ManagedContainer<TracerParticle> tracers(10000);
  for (int k = 0; k < 10000; ++k) {
    tracers.sweep(1000);

    for (int pidx = 0; pidx < 1000; ++pidx) {
      tracers.insert();
    }
    for (auto &tracer : tracers) {
      float t = drand48();
      advect(&tracer);
      if (t > 0.8)
        tracer.isActive = false;
    }
  }

  std::cout << tracers.size() << " " << tracers.get_nbr_active() << std::endl;

  auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  std::chrono::duration<float> elapsed = finish - start;
  std::cout << "elapsed time: " << elapsed.count() << " s\n";
}
