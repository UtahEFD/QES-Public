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
  for (auto parItr = tracer->begin(); parItr != tracer->end(); ++parItr) {
    if (parItr->isRogue) {
      isRogueCount = isRogueCount + 1;
    }
  }
  for (auto parItr = heavy->begin(); parItr != heavy->end(); ++parItr) {
    if (parItr->isRogue) {
      isRogueCount = isRogueCount + 1;
    }
  }
  return isRogueCount;
}

int ParticleContainers::get_nbr_active() const
{
  return tracer->get_nbr_active()
         + heavy->get_nbr_active();
}

int ParticleContainers::get_nbr_active(const ParticleType &in) const
{
  switch (in) {
  case ParticleType::tracer:
    return tracer->get_nbr_active();
  case ParticleType::heavy:
    return heavy->get_nbr_active();
  default:
    exit(1);
  }
}


int ParticleContainers::get_nbr_inserted() const
{
  return tracer->get_nbr_inserted()
         + heavy->get_nbr_inserted();
}
int ParticleContainers::get_nbr_inserted(const ParticleType &in) const
{
  switch (in) {
  case ParticleType::tracer:
    return tracer->get_nbr_inserted();
  case ParticleType::heavy:
    return heavy->get_nbr_inserted();
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
  tracer->sweep(nbr_particle_to_add[ParticleType::tracer]);
  nbr_particle_to_add[ParticleType::tracer] = 0;
  heavy->sweep(nbr_particle_to_add[ParticleType::heavy]);
  nbr_particle_to_add[ParticleType::heavy] = 0;
}


void ParticleContainers::container_info() const
{
  std::cout << "Particle Container info" << std::endl;
  std::cout << "tracer:          " << tracer->size() << " " << tracer->get_nbr_active() << std::endl;
  std::cout << "heavy particles: " << heavy->size() << " " << heavy->get_nbr_active() << std::endl;
}
