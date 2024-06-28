/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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

/** @file RandomSingleton
 * @brief This class handles the random number generation
 */

#include "RandomSingleton.h"

RandomSingleton *RandomSingleton::m_the_instance = nullptr;

RandomSingleton *RandomSingleton::getInstance()
{
  if (m_the_instance == nullptr) {
    m_the_instance = new RandomSingleton();
  }
  return m_the_instance;
}

RandomSingleton::RandomSingleton()
  : m_normal_value(false), m_remaining_value(0.0),
    prng(), distribution(0.0, 1.0)
{
  m_normal_value = false;

  prng.seed(std::time(nullptr));
}

double RandomSingleton::uniRan()
{
  return distribution(prng);
}

double RandomSingleton::norRan()
{
  float rsq, v1, v2;
  if (!m_normal_value) {
    do {
      v1 = 2.0f * uniRan() - 1.0f;
      v2 = 2.0f * uniRan() - 1.0f;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0);

    rsq = sqrt((-2.0f * log(rsq)) / rsq);

    m_remaining_value = v2 * rsq;
    m_normal_value = true;

    return v1 * rsq;
  } else {
    m_normal_value = false;
    return m_remaining_value;
  }
}
