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

/** @file Sources.hpp
 * @brief This class contains data and variables that set flags and
 * settngs read from the xml.
 *
 * @note Child of ParseInterface
 * @sa ParseInterface
 */

#pragma once

#include "Particle.h"


class TracerParticle : public Particle
{
public:
  // initializer
  TracerParticle()
    : Particle(ParticleType::tracer)
  {}

  // destructor
  ~TracerParticle()
  {
  }

private:
};


class BasePhysicalProperties
{
public:
  double d;// particle diameter diameter [microns]
  // double d_m;// particle diameter diameter [m]
  double m;// particle mass [g]
  // double m_kg;// particle mass [kg]
  double m_o;// initial particle mass [g]
  // double m_kg_o;// initial particle mass [kg]
  double rho;// density of particle

  // mass decay constant
  double decayConst;

  // decay variables
  double wdecay;// (1 - fraction) particle decayed [0,1]
};

class BaseParticle
{
public:
  BaseParticle()
    : state(INACTIVE)
  {}

  // particle type
  const ParticleType type = tracer;
  // state of the particle
  ParticleState state;
  // id of particle
  uint32_t ID{};

  Metadata metadata;
  Core core;
  BasePhysicalProperties physical;
};
