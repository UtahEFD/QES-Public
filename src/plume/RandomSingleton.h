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

/** @file Random.h
 * @brief This class handles the random number generation
 */

#pragma once

#include <random>

class RandomSingleton
{
private:
  /**
   * The Singleton's constructor should always be private to prevent
   * direct construction calls with the `new` operator
   */

  RandomSingleton();

  static RandomSingleton *m_the_instance;

  bool m_normal_value;
  float m_remaining_value;

  std::default_random_engine prng;

  // We must also create a distribution from which to pull the random numbers
  // we want.  In this case, I would like random integers to be generated
  // uniformly from betwen -10000000 and 10000000
  std::uniform_real_distribution<float> distribution;

public:
  static RandomSingleton *getInstance();

  float uniRan();
  float norRan();
};
