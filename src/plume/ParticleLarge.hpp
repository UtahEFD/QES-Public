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

/** @file ParticleLarge.hpp
 * @brief Derived from Particle.hpp. Large particles have diameter, mass, settling, density, deposition, and drag
 * (though drag isn't implemented yet, LDU 11/16/21)
 */

#pragma once

#include "util/Vector3.h"
#include "Particle.hpp"

class ParticleLarge : public Particle
{
  friend class ParseParticleLarge;

public:
  // initializer
  ParticleLarge()
    : Particle(true, ParticleType::large)
  {
    //  ParseParticle(const bool &flag, std::string str, const ParticleType &type)
    //    : d(0.0), d_m(0.0), m(0.0), m_kg(0.0), rho(0.0),
    //      depFlag(flag), decayConst(0.0), c1(2.049), c2(1.19),
    //      tag(std::move(str)), particleType(type)
  }

  // initializer
  ParticleLarge(const double &d_part, const double &m_part, const double &rho_part)
    : Particle(true, ParticleType::large)
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
    //tag = "ParticleLarge";

    // (1 - fraction) particle deposited
    depFlag = true;

    // (1 - fraction) particle deposited
    wdecay = 1.0;
  }

  // destructor
  ~ParticleLarge() override = default;

private:
};

class ParseParticleLarge : public ParseParticle
{
protected:
public:
  // default constructor
  ParseParticleLarge() : ParseParticle(true, ParticleType::large)
  {
  }

  // destructor
  ~ParseParticleLarge() = default;

  void parseValues() override
  {
    parsePrimitive<double>(true, rho, "particleDensity");
    parsePrimitive<double>(true, d, "particleDiameter");
    parsePrimitive<bool>(true, depFlag, "depositionFlag");
    parsePrimitive<double>(true, decayConst, "decayConst");
    parsePrimitive<double>(false, c1, "c1");
    parsePrimitive<double>(false, c2, "c2");
    d_m = d * (1.0E-6);
    m_kg = m * (1.0E-3);
  }

  void setParticleParameters(Particle *ptr) override
  {
    auto *tmp = dynamic_cast<ParticleLarge *>(ptr);
    tmp->d = d;
    tmp->d_m = (1.0E-6) * d;
    tmp->rho = rho;
    tmp->depFlag = depFlag;
    tmp->decayConst = decayConst;
    tmp->c1 = c1;
    tmp->c2 = c2;
  }
};
