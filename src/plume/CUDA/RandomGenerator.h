/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file
 * @brief
 */

#ifndef __CUDA_RANDOMGENERATOR_H__
#define __CUDA_RANDOMGENERATOR_H__

#include <cmath>
#include <map>
#include <utility>

#include <cuda.h>
#include <curand.h>

#include "util/VectorMath.h"

typedef struct
{
  int length;
  float *vals;
} rng_array;

class RandomGenerator
{
public:
  RandomGenerator()
  {
    // Create pseudo-random number generator
    // CURAND_CALL(
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);

    // Set the seed --- not sure how we'll do this yet in general
    // CURAND_CALL(
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  }

  ~RandomGenerator()
  {
    // std::cout << "RNG KABOOM" << std::endl;
    freeAll();
    // Cleanup
    curandDestroyGenerator(gen);
  }

  float *get(const std::string &key) { return arrays[key].vals; }

  void create(const std::string &, const int &length);
  void generate(const std::string &, const float &, const float &);
  void destroy(const std::string &);

private:
  void freeAll();

  curandGenerator_t gen;

  std::map<std::string, rng_array> arrays;
};

#endif
