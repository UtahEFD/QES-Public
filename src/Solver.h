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

/** @file Solver.h */

#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>
#include <limits>

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

#include "Vector3.h"

using namespace std;

/**
 * @class Solver
 * @brief Declares and defines variables required for both solvers.
 *
 * An abstract class that is the basis for the windfield convergence algorithm and
 * contains information needed to run the simulation as well as functions widely
 * used by different solver methods. There are several special member variables that
 * should be accessible to all solvers. They are declared in this class.
 *
 * @sa CPUSolver
 * @sa DynamicParallelism
 * @sa GlobalMemory
 * @sa SharedMemory
 */
class Solver
{
protected:
  const int alpha1; /**< Gaussian precision moduli */
  const int alpha2; /**< Gaussian precision moduli */
  const float eta; /**< :document this: */
  const float A; /**< :document this: */
  const float B; /**< :document this: */

  float tol; /**< Error tolerance */
  const float omega = 1.78f; /**< Over-relaxation factor */

  // SOLVER-based parameters
  std::vector<float> R; /**< Divergence of initial velocity field */
  std::vector<float> lambda, lambda_old; /**< :document these as group or indiv: */

  int itermax; /**< Maximum number of iterations */

  /**
   * Prints out the current amount that a process
   * has finished with a progress bar.
   *
   * @param percentage the amount the task has finished
   */
  void printProgress(float percentage);


public:
  Solver(const WINDSInputData *WID, WINDSGeneralData *WGD);

  virtual void solve(const WINDSInputData *WID, WINDSGeneralData *WGD, bool solveWind) = 0;
};
