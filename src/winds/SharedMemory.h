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

/** @file SharedMemory.h */

#pragma once

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "WINDSInputData.h"
#include "Solver.h"


/**
 * @class SharedMemory
 * @brief Child class of the Solver that runs the convergence
 * algorithm using DynamicParallelism on a single GPU.
 *
 * @sa Solver
 * @sa DynamicParallelism
 * @sa WINDSInputData
 */
class SharedMemory : public Solver
{
private:
  /**
   * :document this:
   *
   * @param e :document this:
   * @param func :document this:
   * @param call :document this:
   * @param line :document this:
   */
  template<typename T>
  void _cudaCheck(T e, const char *func, const char *call, const int line);

public:
  SharedMemory(const WINDSInputData *WID, WINDSGeneralData *WGD)
    : Solver(WID, WGD)
  {
  }

protected:
  ///@{
  /** Solver coefficient on device (GPU) */
  float *d_e, *d_f, *d_g, *d_h, *d_m, *d_n;
  ///@}
  float *d_R; /**< Divergence of initial velocity field on device (GPU) */
  ///@{
  /** Lagrange multiplier on device (GPU) */
  float *d_lambda, *d_lambda_old;
  ///@}

  /**
   * :document this:
   *
   * @param WID :document this:
   * @param WGD :document this:
   * @param solveWind :document this:
   */
  virtual void solve(const WINDSInputData *WID, WINDSGeneralData *WGD, bool solveWind);
};