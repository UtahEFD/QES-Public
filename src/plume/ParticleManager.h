//
// Created by Fabien Margairaz on 9/21/23.
//

#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>

#include "Particle.hpp"
#include "Particle_Tracer.hpp"
#include "Particle_Heavy.hpp"

#include "util/ManagedContainer.h"


struct ParticleManager
{
public:
  virtual int get_nbr_rogue() = 0;
  virtual int get_nbr_active() = 0;
  virtual int get_nbr_inserted() = 0;
  // virtual void prepare(const int &) = 0;
  virtual void sweep(const int &new_part) = 0;
};

class TracerManager : public ParticleManager
{
public:
  TracerManager() = default;

  int get_nbr_active()
  {
    return tracers.get_nbr_active();
  }
  int get_nbr_inserted()
  {
    return tracers.get_nbr_inserted();
  }
  int get_nbr_rogue()
  {
    // now update the isRogueCount
    int isRogueCount = 0;
    for (auto parItr = tracers.begin(); parItr != tracers.end(); ++parItr) {
      if (parItr->isRogue) {
        isRogueCount = isRogueCount + 1;
      }
    }
    return isRogueCount;
  }
  void sweep(const int &new_part)
  {
    tracers.sweep(new_part);
  }

private:
  ManagedContainer<Particle_Tracer> tracers;
};


class ParticleBuffer
{
public:
  std::unordered_map<ParticleType, ParticleManager *, std::hash<int>> particles;

  std::vector<Particle_Tracer> buffer;
  std::queue<size_t> available;

  size_t buffer_size = 0;
  size_t nbr_used = 0;

  ParticleBuffer() = default;
  ParticleBuffer(size_t n)
  {
    particles[ParticleType::tracer] = new TracerManager();

    for (auto pt : particles) {
      // pt.second->at(1);
    }
    buffer.resize(n);
    buffer_size = n;
    for (size_t k = 0; k < n; ++k)
      available.push(k);
  }
  ~ParticleBuffer() = default;

  void check_size(const int &new_part);
  void scrub_buffer();
  size_t next();
  size_t size() { return buffer.size(); }
  void add();
};
