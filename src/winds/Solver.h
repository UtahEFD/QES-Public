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

#include "util/Vector3.h"

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
  const float omega = 1.0f; /**< Over-relaxation factor */

  int itermax; /**< Maximum number of iterations */

  // SOLVER-based parameters
  std::vector<float> R; /**< Divergence of initial velocity field */
  std::vector<float> lambda, lambda_old; /**< :document these as group or indiv: */

  Solver(const WINDSInputData *WID, WINDSGeneralData *WGD);
  /**
   * Prints out the current amount that a process
   * has finished with a progress bar.
   *
   * @param percentage the amount the task has finished
   */
  void printProgress(float percentage);


public:
  void resetLambda();
  void copyLambda();

  virtual void solve(const WINDSInputData *WID, WINDSGeneralData *WGD, bool solveWind) = 0;
};

inline void Solver::resetLambda()
{
  std::fill(lambda.begin(), lambda.end(), 0.0);
  std::fill(lambda_old.begin(), lambda_old.end(), 0.0);
  std::fill(R.begin(), R.end(), 0.0);
}
