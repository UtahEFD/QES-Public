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


class ParticleManager
{
public:
  virtual Particle *at(size_t) = 0;
};

class TracerManager : public ManagedContainer<Particle_Tracer>
  , public ParticleManager
{
public:
  Particle *at(size_t k)
  {
    return &elements[k];
  }

private:
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
      pt.second->at(1);
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
