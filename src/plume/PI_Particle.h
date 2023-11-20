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

#include "PI_Source.h"
#include "Particle.hpp"

class Source;
class GroundDeposition;
class CanopyDeposition;

class PI_Particle : public ParseInterface
{
private:
  // default constructor
  PI_Particle()
    : d(0.0), m(0.0), rho(0.0),
      depFlag(false), decayConst(0.0), c1(2.049), c2(1.19)
  {}

protected:
  PI_Particle(const ParticleType &type, const bool &flag)
    : particleType(type),
      d(0.0), m(0.0), rho(0.0),
      depFlag(flag), decayConst(0.0), c1(2.049), c2(1.19)
  {}

public:
  // particle type
  ParticleType particleType;

  std::vector<PI_Source *> sources;

  // Physical properties
  // diameter of particle (micron)
  double d;
  // mass of particle (g)
  double m;
  // density of particle (kg/m3)
  double rho;

  bool depFlag;
  double decayConst, c1, c2;

  // destructor
  ~PI_Particle() = default;

  virtual void parseValues() = 0;

  // virtual void setParticleParameters(Particle *) = 0;
};