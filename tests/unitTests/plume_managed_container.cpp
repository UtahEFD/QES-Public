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

/** @file plume_managed_container.cpp
 * @brief This is a test and example on how the managed container and companion container work
 */

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <list>
#include <chrono>

#include "plume/Particle.h"
#include "plume/ManagedContainer.h"
#include "plume/ParticleIDGen.h"
#include "plume/Random.h"

TEST_CASE("ManagedContainer", "[done]")
{
  auto start = std::chrono::high_resolution_clock::now();
  ManagedContainer<ParticleControl> control;
  std::vector<uint32_t> testID;
  ParticleIDGen *id_gen = ParticleIDGen::getInstance();

  int new_particle = 1E3;
  std::vector<size_t> newIdx;

  for (int n = 0; n < 10000; ++n) {
    control.check_resize(new_particle);
    control.resize_companion(testID);

    control.obtain_available(new_particle, newIdx);

    for (size_t k = 0; k < newIdx.size(); ++k) {
      control[newIdx[k]].reset();
    }
    for (size_t k = 0; k < newIdx.size(); ++k) {
      testID[newIdx[k]] = id_gen->get();
    }

    Random prng;
    for (auto &p : control) {
      float t = prng.uniRan();
      if (t > 0.8)
        p.state = INACTIVE;
    }
  }

  std::cout << control.size() << " " << control.get_nbr_active() << std::endl;

  auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  std::chrono::duration<float> elapsed = finish - start;
  std::cout << "elapsed time: " << elapsed.count() << " s\n";

  REQUIRE(testID[control.get_last_index_added()] == 10000 * new_particle - 1);
  REQUIRE(control.size() == testID.size());
  REQUIRE(control.size() == 6000);
}
