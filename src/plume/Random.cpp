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

/** @file Random.cpp
 * @brief This class handles the random number generation
 */

#include <random>
#include <ctime>

#include "Random.h"

Random::Random()
  : m_normal_value(false), m_remaining_value(0.0), prng(), distribution(0.0, 1.0)
{
  prng.seed(std::time(nullptr));
}

Random::Random(long seed)
  : m_normal_value(false), m_remaining_value(0.0), prng(), distribution(0.0, 1.0)
{
  prng.seed(seed);
}

double Random::uniRan()
{
  return distribution(prng);
}

double Random::norRan()
{
  double rsq, v1, v2;
  if (!m_normal_value) {
    do {
      v1 = 2.0 * uniRan() - 1.0;
      v2 = 2.0 * uniRan() - 1.0;
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


// random::random()
//{
//   m_normal_value = false;
//   std::cout << "Set  RANDOM seed " << std::endl;
//   //std::srand(1);
//   std::srand(std::time(0));
// }
//
// double random::uniRan()
//{
//   return std::rand() / (double)RAND_MAX;
// }
//
// double random::norRan()
//{
//   float rsq, v1, v2;
//   if (m_normal_value == false) {
//     do {
//       v1 = 2.0f * uniRan() - 1.0f;
//       v2 = 2.0f * uniRan() - 1.0f;
//       rsq = v1 * v1 + v2 * v2;
//     } while (rsq >= 1.0);
//
//     rsq = sqrt((-2.0f * log(rsq)) / rsq);
//
//     m_remaining_value = v2 * rsq;
//     m_normal_value = true;
//
//     return v1 * rsq;
//   } else {
//     m_normal_value = false;
//     return m_remaining_value;
//   }
// }
//
// double random::rangen()
//{
//
//   double x, u, v, w, summ;
//
//   u = uniRan();
//   if (u <= .8638) {
//     v = 2.0 * uniRan() - 1.0;
//     w = 2.0 * uniRan() - 1.0;
//     x = 2.315351 * u - 1.0 + v + w;
//     return x;
//   }
//
//   if (u <= .9745) {
//     v = uniRan();
//     x = 1.5 * (v - 1.0 + 9.033424 * (u - .8638));
//     return x;
//   }
//   if (u > .9973002) {
//     v = 4.5;
//     x = 1.0;
//     while ((x * v * v) > 4.5) {
//       v = uniRan();
//       w = uniRan();
//       if (w < 1.0e-7)
//         w = 1.e-07;
//       x = 4.5 - log(w);
//     }
//     x = (sqrt(2.0 * x), (u - .9986501)) / fabs((sqrt(2.0 * x), (u - .9986501)));
//   } else {
//     u = 50.;
//     v = 0.;
//     summ = 49.00244;
//     w = 0.;
//     while (u > (49.00244 * exp(-v * v / 2.0) - summ - w)) {
//       x = 6.0 * uniRan() - 3.0;
//       u = uniRan();
//       v = abs(x);
//       w = pow((6.631334 * (3.0 - v)), 2.0);
//       summ = 0.;
//       if (v < 1.5) summ = 6.043281 * (1.5 - v);
//       if (v < 1.0) summ = summ + 13.26267 * (3.0 - v * v) - w;
//     }
//   }
//   return x;
// }
//
//
// bool random::m_normal_value;
// double random::m_remaining_value;
