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
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file TURBWallTerrain.h */

#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <limits>

#include "WINDSGeneralData.h"
#include "TURBGeneralData.h"
#include "TURBWall.h"


/**
 * @class TURBWallTerrain
 * @brief :document this:
 *
 * @sa TURBWall
 */
class TURBWallTerrain : public TURBWall
{
protected:
  void (TURBWallTerrain::*comp_wall_velocity_deriv)(WINDSGeneralData *, TURBGeneralData *);
  void (TURBWallTerrain::*comp_wall_stress_deriv)(WINDSGeneralData *,
                                                  TURBGeneralData *,
                                                  const std::vector<float> &,
                                                  const std::vector<float> &,
                                                  const std::vector<float> &);

  void set_loglaw_stairstep(WINDSGeneralData *, TURBGeneralData *);

public:
  TURBWallTerrain(const WINDSInputData *, WINDSGeneralData *, TURBGeneralData *);
  ~TURBWallTerrain()
  {}

  void setWallsVelocityDeriv(WINDSGeneralData *, TURBGeneralData *);
  void setWallsStressDeriv(WINDSGeneralData *,
                           TURBGeneralData *,
                           const std::vector<float> &,
                           const std::vector<float> &,
                           const std::vector<float> &);

private:
  TURBWallTerrain()
  {}

  const int icellflag_terrain = 2;
  const int icellflag_cutcell = 8;

  const int iturbflag_stairstep = 2;
  const int iturbflag_cutcell = 3;

  bool use_cutcell = false;
};