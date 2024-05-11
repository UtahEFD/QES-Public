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

/** @file TURBWallBuilding.h */

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
 * @class TURBWallBuilding
 * @brief :document this:
 *
 * @sa TURBWall
 */
class TURBWallBuilding : public TURBWall
{
protected:
  void (TURBWallBuilding::*comp_wall_velocity_deriv)(WINDSGeneralData *, TURBGeneralData *);
  void (TURBWallBuilding::*comp_wall_stress_deriv)(WINDSGeneralData *,
                                                   TURBGeneralData *,
                                                   const std::vector<float> &,
                                                   const std::vector<float> &,
                                                   const std::vector<float> &);

  void set_loglaw_stairstep(WINDSGeneralData *, TURBGeneralData *);

public:
  TURBWallBuilding(const WINDSInputData *, WINDSGeneralData *, TURBGeneralData *);
  ~TURBWallBuilding()
  {}

  void setWallsVelocityDeriv(WINDSGeneralData *, TURBGeneralData *);
  void setWallsStressDeriv(WINDSGeneralData *,
                           TURBGeneralData *,
                           const std::vector<float> &,
                           const std::vector<float> &,
                           const std::vector<float> &);

private:
  TURBWallBuilding()
  {}

  const int icellflag_building = 0;
  const int icellflag_cutcell = 7;

  const int iturbflag_stairstep = 4;
  const int iturbflag_cutcell = 5;

  bool use_cutcell = false;
};
