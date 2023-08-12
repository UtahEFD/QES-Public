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

/** @file ParticleTracer.hpp 
 * @brief Derived from Particle.hpp. Tracer particles are massless and do not settle, deposit, or experience drag effects.
 */

#pragma once

#include "util/Vector3.h"
#include "Particle.hpp"

class ParticleTracer : public Particle
{

public:
  // initializer
  ParticleTracer()
  {
    // diameter of particle (micron and m)
    d = 0.0;
    d_m = (1.0E-6) * d;

    // mass of particle (g and kg)
    m = 0.0;
    m_kg = (1.0E-3) * m;

    // density of particle
    rho = 0.0;

    // tag
    tag = "ParticleTracer";

    // (1 - fraction) particle deposited
    depFlag = false;

    // (1 - fraction) particle decay
    wdecay = 1.0;
  }

  // initializer
  ParticleTracer(const double &d_part, const double &m_part, const double &rho_part)
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

  /*  void parseValues()
  {
      parType = ParticleType::tracer;
      
  }
*/

  //  void setSettlingVelocity(const double &, const double &){
  //    vs = 0.0;
  void setSettlingVelocity(const double &rhoAir, const double &nuAir)
  {
    vs = 0.0;
  }

private:
};
