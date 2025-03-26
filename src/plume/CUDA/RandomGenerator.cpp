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

/** @file
 * @brief
 */

#include "RandomGenerator.h"

void RandomGenerator::freeAll()
{
  for (auto &[key, array] : arrays) {
    // std::cout << '[' << key << "]" << std::endl;
    cudaFree(array.vals);
  }
  arrays.clear();
}

void RandomGenerator::create(const std::string &key, const int &length)
{
  // float *d_tmp;
  arrays[key].length = length;
  cudaMalloc((void **)&arrays[key].vals, length * sizeof(float));
  //  arrays[key] = { length, d_tmp };
}

void RandomGenerator::generate(const std::string &key, const float &m, const float &s)
{
  if (arrays.count(key) == 1) {
    curandGenerateNormal(gen, arrays[key].vals, arrays[key].length, m, s);
  } else {
    std::cerr << "ERROR KEY DOES NOT EXIST" << std::endl;
    exit(1);
  }
}

void RandomGenerator::destroy(const std::string &key)
{
  if (arrays.count(key) == 1) {
    cudaFree(arrays[key].vals);
    arrays.erase(key);
  }
}
