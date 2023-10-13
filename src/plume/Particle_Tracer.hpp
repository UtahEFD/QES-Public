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
#include "Particle.hpp"


class Particle_Tracer : public Particle
{
  friend class ParseParticle_Tracer;

public:
  // initializer
  Particle_Tracer()
    : Particle(false, ParticleType::tracer)
  {
    //  ParseParticle(const bool &flag, std::string str, const ParticleType &type)
    //    : d(0.0), d_m(0.0), m(0.0), m_kg(0.0), rho(0.0),
    //      depFlag(flag), decayConst(0.0), c1(2.049), c2(1.19),
    //      tag(std::move(str)), particleType(type)
  }

  explicit Particle_Tracer(const size_t &ID)
    : Particle(false, ParticleType::tracer)
  {
    isActive = true;
    particleID = ID;
  }

  // initializer
  Particle_Tracer(const double &d_part, const double &m_part, const double &rho_part)
    : Particle(false, ParticleType::tracer)
  {
    // diameter of particle (micron and m)
    // d = d_part;
    // d_m = (1.0E-6) * d;

    // mass of particle (g and kg)
    // m = m_part;
    // m_kg = (1.0E-3) * m;

    // density of particle
    // rho = rho_part;

    // tag
    // tag = "Particle_Tracer";

    // (1 - fraction) particle deposited
    depFlag = false;

    // (1 - fraction) particle deposited
    wdecay = 1.0;
  }

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