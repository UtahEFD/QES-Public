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

/** @file ParticleHeavyGas.hpp
 * @brief Derived from Particle.hpp. Heavy gas attributes not defined yet,
 * but this script is identical to Particle_Heavy and ParticleLarge.
 */

#pragma once

#include "util/Vector3.h"
#include "Particle.hpp"

class ParticleHeavyGas : public Particle
{
  friend class ParseParticleHeavyGas;

public:
  // initializer
  ParticleHeavyGas()
    : Particle(true, "ParticleHeavyGas", ParticleType::heavygas)
  {
    //  ParseParticle(const bool &flag, std::string str, const ParticleType &type)
    //    : d(0.0), d_m(0.0), m(0.0), m_kg(0.0), rho(0.0),
    //      depFlag(flag), decayConst(0.0), c1(2.049), c2(1.19),
    //      tag(std::move(str)), particleType(type)
  }

  // initializer
  ParticleHeavyGas(const double &d_part, const double &m_part, const double &rho_part)
    : Particle(true, "ParticleHeavyGas", ParticleType::heavygas)
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
    tag = "ParticleHeavyGas";

    // (1 - fraction) particle deposited
    depFlag = true;

    // (1 - fraction) particle deposited
    wdecay = 1.0;
  }

  // destructor
  ~ParticleHeavyGas()
  {
  }

  //  void setSettlingVelocity(const double &, const double &);
  void setSettlingVelocity(const double &rhoAir, const double &nuAir) override
  {
    if (d > 0 && rho > rhoAir) {
      // dimensionless grain diameter
      dstar = d_m * pow(9.81 / pow(nuAir, 2.0) * (rho / rhoAir - 1.), 1.0 / 3.0);
      // drag coefficent
      Cd = (432.0 / pow(dstar, 3.0)) * pow(1.0 + 0.022 * pow(dstar, 3.0), 0.54)
           + 0.47 * (1.0 - exp(-0.15 * pow(dstar, 0.45)));
      // dimensionless settling velociy
      wstar = pow((4.0 * dstar) / (3.0 * Cd), 0.5);
      // settling velocity
      vs = wstar * pow(9.81 * nuAir * (rho / rhoAir - 1.0), 1.0 / 3.0);
    } else {
      vs = 0.0;
    }
  }

private:
};

class ParseParticleHeavyGas : public ParseParticle
{
protected:
public:
  // default constructor
  ParseParticleHeavyGas()
    : ParseParticle(true, "ParticleHeavyGas", ParticleType::heavygas)
  {
  }

  // destructor
  ~ParseParticleHeavyGas()
  {
  }

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
    auto *tmp = dynamic_cast<ParticleHeavyGas *>(ptr);
    tmp->d = 0.0;
    tmp->d_m = (1.0E-6) * 0.0;
    tmp->rho = rho;
    tmp->depFlag = depFlag;
    tmp->decayConst = decayConst;
    tmp->c1 = c1;
    tmp->c2 = c2;
  }
};