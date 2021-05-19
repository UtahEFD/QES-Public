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

/** @file CPUSolver.h */

#pragma once

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>

#include "WINDSInputData.h"
#include "Solver.h"

/**
 * @class CPUSolver
 * @brief Child class of the Solver that runs the convergence
 * algorithm in serial order on a CPU.
 */
class CPUSolver : public Solver
{
public:
  CPUSolver(const WINDSInputData *WID, WINDSGeneralData *WGD)
    : Solver(WID, WGD)
  {
  }

protected:
  /** :document this:
   * Start by writing a one sentence description here
   *
   * Document the implementation details in the .cpp file, not here.
   * (remove the placeholder comments and :document this: tag when done)
   *
   * @param WID :document this:
   * @param WGD :document this:
   * @param solveWind :document this:
   */
  virtual void solve(const WINDSInputData *WID, WINDSGeneralData *WGD, bool solveWind);
};
