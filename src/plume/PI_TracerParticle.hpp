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

#include "util/ParseInterface.h"

#include "PI_Particle.hpp"

#include "ParticleModel.h"
#include "TracerParticle_Model.h"

class PI_TracerParticle : public PI_Particle
{
protected:
public:
  // default constructor
  PI_TracerParticle()
    : PI_Particle(ParticleType::tracer, false)
  {}

  // destructor
  ~PI_TracerParticle()
  {
  }

  void parseValues() override
  {
    parsePrimitive<std::string>(true, tag, "tag");

    parsePrimitive<float>(false, rho, "particleDensity");
    parsePrimitive<float>(false, d, "particleDiameter");

    parsePrimitive<bool>(false, depFlag, "depositionFlag");
    parsePrimitive<float>(false, c1, "c1");
    parsePrimitive<float>(false, c2, "c2");

    parsePrimitive<float>(false, decayConst, "decayConst");

    parseMultiElements(false, sources, "source");
  }

  void initialize(const PI_PlumeParameters *plumeParams) override
  {
    for (auto s : sources) {
      s->initialize(plumeParams->timeStep);
    }
  }

  virtual ParticleModel *create(QESDataTransport &data) override
  {
    auto *model = new TracerParticle_Model(data, tag);
    for (auto s : sources) {
      // add source into the vector of sources
      model->addSource(s->create(data));
    }
    return model;
  }

  // void setParticleParameters(Particle *ptr) override {}
};