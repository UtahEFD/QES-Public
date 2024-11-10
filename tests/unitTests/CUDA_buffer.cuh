
#pragma once

#include "Particle.cuh"
#include "util/VectorMath_CUDA.h"

typedef struct
{
  int N;// Multiple of warp size
  int capacity;// Power of 2
  int *head_h;
  int *size_h;
  size_t *peek_head_h;
  size_t *data_h;
} ring;


__device__ void push_ring(ring *in, const int tid, const size_t val)
{
  // Wrap tail if needed
  int x = in->head_h[tid] + in->size_h[tid];
  x &= in->capacity - 1;
  in->data_h[(x * in->N) + tid] = val;

  // Increase size
  in->size_h[tid]++;
  if (in->size_h[tid] >= in->capacity) {
    // Handle full buffer (optional)
    // ...
  }

  // Set peek head if it's the first element
  if (in->size_h[tid] == 1) {
    in->peek_head_h[tid] = val;
  }
}

__device__ void pop_ring(ring *in, const int tid)
{
  // If empty, return
  if (in->size_h[tid] == 0) {
    return;
  }

  // Advance head (wrap if needed)
  in->head_h[tid]++;
  if (in->head_h[tid] == in->capacity) {
    in->head_h[tid] = 0;
  }

  // Update size and peek head
  in->size_h[tid]--;
  if (in->size_h[tid] > 0) {
    in->peek_head_h[tid] = in->data_h[(in->head_h[tid] * in->N) + tid];
  }
}


static inline void push_ring_host(ring *in, const int tid, const size_t val)
{
  // Wrap tail if needed
  int x = in->head_h[tid] + in->size_h[tid];
  x &= in->capacity - 1;
  in->data_h[(x * in->N) + tid] = val;

  // Increase size
  in->size_h[tid]++;
  if (in->size_h[tid] >= in->capacity) {
    // Handle full buffer (optional)
    // ...
  }

  // Set peek head if it's the first element
  if (in->size_h[tid] == 1) {
    in->peek_head_h[tid] = val;
  }
}

static inline void pop_ring_host(ring *in, const int tid)
{
  // If empty, return
  if (in->size_h[tid] == 0) {
    return;
  }

  // Advance head (wrap if needed)
  in->head_h[tid]++;
  if (in->head_h[tid] == in->capacity) {
    in->head_h[tid] = 0;
  }

  // Update size and peek head
  in->size_h[tid]--;
  if (in->size_h[tid] > 0) {
    in->peek_head_h[tid] = in->data_h[(in->head_h[tid] * in->N) + tid];
  }
}
