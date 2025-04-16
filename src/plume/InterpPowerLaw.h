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

/** @file InterpPowerLaw.h
 * @brief
 */

#pragma once

#include <iostream>
#include <ctime>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "util/calcTime.h"
#include "Random.h"
#include "util/Vector3Float.h"

#include "Interp.h"


class InterpPowerLaw : public Interp
{
public:
  // constructor
  // copies the turb grid values for nx, ny, nz, nt, dx, dy, and dz to the QES grid values,
  // then calculates the tau gradients which are then used to calculate the flux_div grid values.
  InterpPowerLaw(qes::Domain, bool);

  void interpWindsValues(const WINDSGeneralData *WGD,
                         const vec3 &pos,
                         vec3 &vel_out) override;

  void interpTurbValues(const TURBGeneralData *TGD,
                        const vec3 &pos,
                        mat3sym &tau_out,
                        vec3 &flux_div_out,
                        float &nuT_out,
                        float &CoEps_out) override;

  void interpTurbInitialValues(const TURBGeneralData *TGD,
                               const vec3 &pos,
                               mat3sym &tau_out,
                               vec3 &sig_out) override;

  float getMaxFluctuation();
};
