/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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

/** @file WallReflection_StairStep.h
 * @brief
 */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
#include <cstring>

#include "util/QEStime.h"
#include "util/calcTime.h"
#include "util/Vector3Float.h"
#include "Random.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "Interp.h"
#include "InterpNearestCell.h"
#include "InterpPowerLaw.h"
#include "InterpTriLinear.h"

#include "WallReflection.h"


class WallReflection_StairStep : public WallReflection
{
public:
  WallReflection_StairStep(Interp *interp)
    : WallReflection(interp)
  {}
  ~WallReflection_StairStep()
  {}

  virtual void reflect(const WINDSGeneralData *WGD, vec3 &pos, vec3 &dist, vec3 &fluct, ParticleState &state) override;

private:
  void trajectorySplit_recursive(const WINDSGeneralData *,
                                 Vector3Float &,
                                 Vector3Float &,
                                 const float &,
                                 float &,
                                 float &,
                                 Vector3Float &,
                                 bool &);

  void oneReflection(const WINDSGeneralData *,
                     Vector3Float &,
                     Vector3Float &,
                     const float &,
                     Vector3Float &,
                     bool &);
};
