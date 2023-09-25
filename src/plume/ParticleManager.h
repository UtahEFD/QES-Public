//
// Created by Fabien Margairaz on 9/21/23.
//

#pragma once

#include <iostream>
#include <vector>
#include <queue>

#include "Particle.hpp"
#include "ParticleTracer.hpp"
#include "ParticleSmall.hpp"
#include "ParticleLarge.hpp"
#include "ParticleHeavyGas.hpp"

class ParticleBuffer
{
public:
  std::vector<ParticleTracer> buffer;
  std::queue<size_t> available;

  size_t buffer_size = 0;
  size_t nbr_used = 0;

  ParticleBuffer() = default;
  ParticleBuffer(size_t n)
  {
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

template<class T>
class ParticleManager
{
public:
  std::vector<T> buffer;
  std::queue<size_t> available;
  std::vector<size_t> added;
  size_t nbr_used = 0;

  ParticleManager() = default;
  explicit ParticleManager(size_t n) : buffer(n)
  {
    for (size_t k = 0; k < n; ++k)
      available.push(k);
  }
  ~ParticleManager() = default;

  size_t front() { return available.front(); }
  void pop() { available.pop(); }
  size_t size() { return buffer.size(); }
  size_t active() { return nbr_used; }
  
  void check_size(const int &new_part)
  {
    added.clear();
    size_t buffer_size = buffer.size();
    if (buffer_size < nbr_used + new_part) {
      buffer.resize(buffer_size + new_part);
      scrub_buffer();
    }
  }

  void scrub_buffer()
  {
    nbr_used = 0;
    std::queue<size_t> empty;
    std::swap(available, empty);
    added.clear();

    for (size_t it = 0; it < buffer.size(); ++it) {
      if (buffer[it].isActive)
        nbr_used++;
      else
        available.push(it);
    }
  }

  void add()
  {
    buffer[available.front()] = T();
    // buffer[available.front()].isActive = true;
    added.emplace_back(available.front());
    available.pop();
  }
};
