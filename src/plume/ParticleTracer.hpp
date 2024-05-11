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

/** @file ParticleTracer.hpp
 * @brief Derived from Particle.hpp. Tracer particles are massless and do not settle, deposit,
 * or experience drag effects.
 */

#pragma once

#include "util/Vector3.h"
#include "Particle.hpp"


class ParticleTracer : public Particle
{
  friend class ParseParticleTracer;

public:
  // initializer
  ParticleTracer()
    : Particle(false, "ParticleTracer", ParticleType::tracer)
  {
    //  ParseParticle(const bool &flag, std::string str, const ParticleType &type)
    //    : d(0.0), d_m(0.0), m(0.0), m_kg(0.0), rho(0.0),
    //      depFlag(flag), decayConst(0.0), c1(2.049), c2(1.19),
    //      tag(std::move(str)), particleType(type)
  }

  // initializer
  ParticleTracer(const double &d_part, const double &m_part, const double &rho_part)
    : Particle(false, "ParticleTracer", ParticleType::tracer)
  {
    // diameter of particle (micron and m)
    d = d_part;
    d_m = (1.0E-6) * d;

    // mass of particle (g and kg)
    m = m_part;
    m_kg = (1.0E-3) * m;

    // density of particle
    rho = rho_part;

    // tag
    tag = "ParticleTracer";

    // (1 - fraction) particle deposited
    depFlag = false;

    // (1 - fraction) particle deposited
    wdecay = 1.0;
  }

  // destructor
  ~ParticleTracer()
  {
  }

  //  void setSettlingVelocity(const double &, const double &){
  //    vs = 0.0;
  void setSettlingVelocity(const double &rhoAir, const double &nuAir) override
  {
    vs = 0.0;
  }

private:
};

class ParseParticleTracer : public ParseParticle
{
protected:
public:
  // default constructor
  ParseParticleTracer()
    : ParseParticle(false, "ParticleTracer", ParticleType::tracer)
  {}

  // destructor
  ~ParseParticleTracer()
  {
  }

  void parseValues() override
  {
  }

  void setParticleParameters(Particle *ptr) override
  {
  }
};