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

struct pairCellFaceID
{
  int cellID;
  int faceID;
};

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

  virtual void setWallsVelocityDeriv(WINDSGeneralData *, TURBGeneralData *) = 0;
  virtual void setWallsStressDeriv(WINDSGeneralData *,
                                   TURBGeneralData *,
                                   const std::vector<float> &,
                                   const std::vector<float> &,
                                   const std::vector<float> &) = 0;

protected:
  void get_stairstep_wall_id(WINDSGeneralData *, int);
  void set_stairstep_wall_flag(TURBGeneralData *, int);

  void get_cutcell_wall_id(WINDSGeneralData *, int);
  void set_cutcell_wall_flag(TURBGeneralData *, int);

  void set_loglaw_stairstep_at_id_cc(WINDSGeneralData *, TURBGeneralData *, int, int, float);

  virtual void set_loglaw_stairstep(WINDSGeneralData *, TURBGeneralData *) = 0;
  void comp_velocity_deriv_finitediff_stairstep(WINDSGeneralData *, TURBGeneralData *);
  void comp_stress_deriv_finitediff_stairstep(WINDSGeneralData *,
                                              TURBGeneralData *,
                                              const std::vector<float> &,
                                              const std::vector<float> &,
                                              const std::vector<float> &);

  // cell
  std::vector<pairCellFaceID> wall_right_indices; /**< Indices of the cells with wall to right boundary condition */
  std::vector<pairCellFaceID> wall_left_indices; /**< Indices of the cells with wall to left boundary condition */
  std::vector<pairCellFaceID> wall_above_indices; /**< Indices of the cells with wall above boundary condition */
  std::vector<pairCellFaceID> wall_below_indices; /**< Indices of the cells with wall bellow boundary condition */
  std::vector<pairCellFaceID> wall_back_indices; /**< Indices of the cells with wall in back boundary condition */
  std::vector<pairCellFaceID> wall_front_indices; /**< Indices of the cells with wall in front boundary condition */

  // cells above wall (for stair-step methods for Wall BC)
  std::vector<int> stairstep_wall_id;
  // cut-cell cells
  std::vector<int> cutcell_wall_id;

private:
};
