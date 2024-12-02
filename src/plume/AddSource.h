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

/** @file Particle.h
 * @brief This class represents information stored for each particle
 */

#pragma once

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "ParticleModel_Visitor.h"

#include "TracerParticle_Model.h"
#include "TracerParticle_Source.h"

// #include "HeavyParticle_Model.h"
// #include "HeavyParticle_Source.h"

class AddSource : public ParticleModel_Visitor
{
public:
  AddSource(std::vector<TracerParticle_Source *> sources)
    : m_tracerParticle_sources(sources)//, m_heavyParticle_sources(0)
  {
  }

  /*AddSource(std::vector<HeavyParticle_Source *> sources)
    : m_tracerParticle_sources(0), m_heavyParticle_sources(sources)
  {
  }*/

  ~AddSource() = default;

  void visit(TracerParticle_Model *element) override
  {
    element->addSources(m_tracerParticle_sources);
  }
  /*void visit(HeavyParticle_Model *element) override
  {
    element->addSources(m_heavyParticle_sources);
    }*/

private:
  AddSource()
  {}

  std::vector<TracerParticle_Source *> m_tracerParticle_sources;
  // std::vector<HeavyParticle_Source *> m_heavyParticle_sources;
};
