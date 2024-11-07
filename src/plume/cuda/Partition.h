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

#include <cmath>
#include <utility>

#include "util/VectorMath.h"
#include "Particle.h"

class Partition
{
public:
  Partition(const int &s) : m_length(s)
  {
    allocate_device();
  }
  ~Partition()
  {
    // std::cout << "call partition KABOOM" << std::endl;
    free_device();
  }

  int active() { return h_lower_count; }
  int empty() { return h_upper_count; }
  int length() { return m_length; }

  void allocate_device_particle_list(particle_array &d_particle_list, const int &length);
  void free_device_particle_list(particle_array &d_particle_list);


  int run(int k, particle_array d_particle[]);
  void insert(int new_particle, particle_array d_new_particle, particle_array d_particle);
  void check(particle_array d_particle, int &active, int &empty);

private:
  Partition() : m_length(0) {}

  void allocate_device();
  void free_device();

  int m_length;
  int m_gpuID = 0;

  int h_lower_count = 0, h_upper_count = 0;
  int *d_lower_count, *d_upper_count;

  int *d_sorting_index;
};
