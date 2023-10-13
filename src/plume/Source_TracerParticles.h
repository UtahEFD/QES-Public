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

/** @file Source_Tracer.hpp
 * @brief  This class represents a generic source type
 */

#pragma once

#include <random>
#include <list>

#include "Source.hpp"

#include "Particle.hpp"
#include "Particle_Tracer.hpp"

#include "ParticleManager.h"
#include "ParticleContainers.h"
#include "ParticleFactories.hpp"

#include "ReleaseType.hpp"
#include "ReleaseType_instantaneous.hpp"
#include "ReleaseType_continuous.hpp"
#include "ReleaseType_duration.hpp"

// #include "Interp.h"
#include "util/ParseInterface.h"
#include "winds/WINDSGeneralData.h"

class Source_TracerParticles : public Source
{
public:
  Source_TracerParticles(const int &sidx, const ParseSource *in) : Source(sidx, in) {}
  // destructor
  ~Source_TracerParticles() override = default;

  int getNewParticleNumber(const float &dt,
                           const float &currTime) override;

  void emitParticles(const float &dt,
                     const float &currTime,
                     ParticleContainers *particles) override;
};
