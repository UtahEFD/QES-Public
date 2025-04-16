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

/** @file GLE_Solver.h
 * @brief
 */

#pragma once

#include "util/VectorMath.h"
#include "winds/TURBGeneralData.h"

#include "Particle.h"
#include "Interp.h"

class PLUMEGeneralData;

class GLE_Solver
{
public:
  void virtual solve(float &,
                     ParticleCore &,
                     ParticleLSDM &,
                     ParticleState &,
                     TURBGeneralData *,
                     PLUMEGeneralData *) = 0;
};


class GLE_Solver_CPU : public GLE_Solver
{
public:
  void solve(float &,
             ParticleCore &,
             ParticleLSDM &,
             ParticleState &,
             TURBGeneralData *,
             PLUMEGeneralData *) override;
};
