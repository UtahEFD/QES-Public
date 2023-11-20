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

/** @file Particle.h
 * @brief This class represents information stored for each particle
 */

#pragma once

#include "util/ManagedContainer.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "Deposition.h"
#include "ParticleContainers.h"
#include "Source.hpp"

class Plume;

class ParticleModel
{
public:
  virtual void generateParticleList(const float &time, const float &dt, WINDSGeneralData *WGD, TURBGeneralData *TGD, Plume *plume) = 0;
  virtual void advect(const double &timeRemainder, WINDSGeneralData *WGD, TURBGeneralData *TGD, Plume *plume) = 0;

  ~ParticleModel() {}

protected:
  ParticleModel(Deposition *dep)
    : deposition(dep)
  {}

  Deposition *deposition = nullptr;

private:
  ParticleModel()
  {}
};
