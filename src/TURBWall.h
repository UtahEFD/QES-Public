/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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

/** @file TURBWall.h */

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

// may need to forward reference this???
class TURBGeneralData;

/**
 * @class TURBWall
 * @brief :document this:
 */
class TURBWall
{
public:
  TURBWall()
  {}
  ~TURBWall()
  {}

  /**
   * Takes in the icellflags set by setCellsFlag
   * function for stair-step method and sets related coefficients to
   * zero to define solid walls. It also creates vectors of indices
   * of the cells that have wall to right/left, wall above/bellow
   * and wall in front/back
   */
  virtual void defineWalls(WINDSGeneralData *, TURBGeneralData *) = 0;
  virtual void setWallsBC(WINDSGeneralData *, TURBGeneralData *) = 0;

protected:
  void get_stairstep_wall_id(WINDSGeneralData *, int);
  void set_stairstep_wall_flag(TURBGeneralData *, int);

  void get_cutcell_wall_id(WINDSGeneralData *, int);
  void set_cutcell_wall_flag(TURBGeneralData *, int);

  void set_loglaw_stairstep_at_id_cc(WINDSGeneralData *, TURBGeneralData *, int, int, float);

  // cells above wall (for stair-step methods for Wall BC)
  std::vector<int> stairstep_wall_id;
  // cut-cell cells
  std::vector<int> cutcell_wall_id;

private:
};
