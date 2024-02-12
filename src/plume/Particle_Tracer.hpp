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

/** @file Particle_Tracer.hpp
 * @brief Derived from Particle.hpp. Tracer particles are massless and do not settle, deposit,
 * or experience drag effects.
 */

#pragma once

#include "util/Vector3.h"
#include "Particle.h"


class Particle_Tracer : public Particle
{
  friend class ParseParticle_Tracer;

public:
  // initializer
  Particle_Tracer()
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
  ~Particle_Tracer()
  {
  }

private:
};

class ParseParticle_Tracer : public ParseParticle
{
protected:
public:
  // default constructor
  ParseParticle_Tracer()
    : ParseParticle(false, ParticleType::tracer)
  {}

  // destructor
  ~ParseParticle_Tracer()
  {
  }

  void parseValues() override
  {
  }

  void setParticleParameters(Particle *ptr) override
  {
  }
};