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

/** @file Sources.hpp
 * @brief This class contains data and variables that set flags and
 * settngs read from the xml.
 *
 * @note Child of ParseInterface
 * @sa ParseInterface
 */

#pragma once

#include "util/ParseInterface.h"

#include "PI_Particle.h"
#include "Particle.hpp"

class PI_TracerParticle : public PI_Particle
{
protected:
public:
  // default constructor
  PI_TracerParticle()
    : PI_Particle(false)
  {}

  // destructor
  ~PI_TracerParticle()
  {
  }

  void parseValues() override
  {
    parseMultiElements(false, sources, "source");
  }

  // void setParticleParameters(Particle *ptr) override {}
};

class TracerParticle : public Particle
{
public:
  // initializer
  TracerParticle()
    : Particle(false, ParticleType::tracer)
  {
  }
  /*
    explicit Particle_Tracer(const size_t &ID)
      : Particle(false, ParticleType::tracer, 0.0, 0.0, 0.0)
    {
      isActive = true;
      particleID = ID;
    }

  // initializer
  Particle_Tracer(const double &d_p, const double &m_p, const double &rho_p)
    : Particle(false, ParticleType::tracer, d_p, m_p, rho_p)
        {
        }
      */
  // destructor
  ~TracerParticle()
  {
  }

private:
};
