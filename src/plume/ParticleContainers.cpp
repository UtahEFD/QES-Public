/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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

/** @file ParticleContainer.cpp
 */

#include "ParticleContainers.h"

int ParticleContainers::get_nbr_rogue() const
{
  // now update the isRogueCount
  int isRogueCount = 0;
  for (const auto &parItr : tracers->elements) {
    if (parItr.isRogue) {
      isRogueCount = isRogueCount + 1;
    }
  }
  for (const auto &parItr : heavy_particles->elements) {
    if (parItr.isRogue) {
      isRogueCount = isRogueCount + 1;
    }
  }
  return isRogueCount;
}

int ParticleContainers::get_nbr_active() const
{
  return tracers->get_nbr_active()
         + heavy_particles->get_nbr_active();
}

int ParticleContainers::get_nbr_active(const ParticleType &in) const
{
  switch (in) {
  case tracer:
    return tracers->get_nbr_active();
  case small:
    return heavy_particles->get_nbr_active();
  case large:
    exit(1);
    break;
  case heavygas:
    exit(1);
    break;
  default:
    exit(1);
  }
}


int ParticleContainers::get_nbr_inserted() const
{
  return tracers->get_nbr_inserted()
         + heavy_particles->get_nbr_inserted();
}
int ParticleContainers::get_nbr_inserted(const ParticleType &in) const
{
  switch (in) {
  case tracer:
    return tracers->get_nbr_inserted();
  case small:
    return heavy_particles->get_nbr_inserted();
  case large:
    exit(1);
    break;
  case heavygas:
    exit(1);
    break;
  default:
    exit(1);
  }
}

void ParticleContainers::prepare(const ParticleType &in, const int &nbr)
{
  nbr_particle_to_add[in] += nbr;
}
void ParticleContainers::sweep()
{
  tracers->sweep(nbr_particle_to_add[ParticleType::tracer]);
  nbr_particle_to_add[ParticleType::tracer] = 0;
  heavy_particles->sweep(nbr_particle_to_add[ParticleType::small]);
  nbr_particle_to_add[ParticleType::small] = 0;
}


void ParticleContainers::container_info() const
{
  std::cout << "Particle Container info" << std::endl;
  std::cout << "tracer:          " << tracers->size() << " " << tracers->get_nbr_active() << std::endl;
  std::cout << "heavy particles: " << heavy_particles->size() << " " << heavy_particles->get_nbr_active() << std::endl;
}
