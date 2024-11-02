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

/** @file Solver_GPU_GlobalMemory.h */

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

#include "Solver.h"


/**
 * @class Solver_GPU_GlobalMemory
 * @brief Child class of the Solver that runs the convergence
 * algorithm using DynamicParallelism on a single GPU.
 *
 * @sa Solver
 * @sa Solver_GPU_GlobalMemory
 */
class Solver_GPU_GlobalMemory : public Solver
{
private:
  /**
   * :document this:
   *
   * @param func :document this:
   * @param call :document this:
   * @param line :document this:
   */
  template<typename T>
  void _cudaCheck(T e, const char *func, const char *call, const int line);

public:
  Solver_GPU_GlobalMemory(qes::Domain domain_in, const float &tolerance);

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
   * @param WGD :document this:
   * @param itermax :document this:
   */
  virtual void solve(WINDSGeneralData *, const int &) override;
};
