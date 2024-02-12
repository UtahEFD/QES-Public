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

/** @file ParticleContainer.h
 */

#pragma once

#include <unordered_map>

#include "Particle.h"
#include "Particle_Tracer.hpp"
#include "Particle_Heavy.hpp"

#include "util/ManagedContainer.h"

class ParticleContainers
{
private:
  std::unordered_map<ParticleType, int, std::hash<int>> nbr_particle_to_add;

public:
  ManagedContainer<Particle_Tracer> *tracer;
  ManagedContainer<Particle_Heavy> *heavy;

  ParticleContainers()
  {
    tracer = new ManagedContainer<Particle_Tracer>();
    nbr_particle_to_add[ParticleType::tracer] = 0;

    heavy = new ManagedContainer<Particle_Heavy>();
    nbr_particle_to_add[ParticleType::heavy] = 0;
  }

  int get_nbr_rogue() const;
  int get_nbr_active() const;
  int get_nbr_active(const ParticleType &) const;
  int get_nbr_inserted() const;
  int get_nbr_inserted(const ParticleType &) const;

  void prepare(const ParticleType &, const int &);

  void sweep();
  void container_info() const;
};
