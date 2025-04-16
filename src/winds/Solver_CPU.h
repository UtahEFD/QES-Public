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

/** @file Solver_CPU.h */

#pragma once

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>

#include "Solver.h"

/**
 * @class Solver_CPU
 * @brief Child class of the Solver that runs the convergence
 * algorithm in serial order on a CPU.
 */
class Solver_CPU : public Solver
{
public:
  Solver_CPU(qes::Domain domain_in, const float &tolerance)
    : Solver(std::move(domain_in), tolerance)
  {
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "[Solver]\t Initializing Serial Solver (CPU) ..." << std::endl;
  }

protected:
  /**
   * :document this:
   *
   * @param WGD :document this:
   * @param itermax :document this:
   */
  virtual void solve(WINDSGeneralData *, const int &) override;
};
