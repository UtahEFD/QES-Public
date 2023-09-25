//
// Created by Fabien Margairaz on 9/21/23.
//

#include "ParticleManager.h"

void ParticleBuffer::check_size(const int &new_part)
{
  buffer_size = buffer.size();

  if (buffer_size < nbr_used + new_part) {
    buffer.resize(buffer_size + new_part);
    scrub_buffer();
    buffer_size = buffer.size();
  }
}

void ParticleBuffer::scrub_buffer()
{
  nbr_used = 0;
  std::queue<size_t> empty;
  std::swap(available, empty);

  for (size_t it = 0; it < buffer.size(); ++it) {
    if (buffer[it].isActive)
      nbr_used++;
    else
      available.push(it);
  }
}

size_t ParticleBuffer::next()
{
  size_t r = available.front();
  available.pop();
  return r;
}
void ParticleBuffer::add()
{
  buffer[available.front()] = ParticleTracer();
  buffer[available.front()].isActive = true;
  available.pop();
}