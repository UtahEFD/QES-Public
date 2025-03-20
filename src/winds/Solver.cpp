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

/**
 * @file Solver.cpp
 * @brief Declares and defines variables required for both solvers.
 *
 * An abstract class that is the basis for the windfield convergence algorithm and
 * contains information needed to run the simulation as well as functions widely
 * used by different solver methods. There are several special member variables that
 * should be accesible to all solvers. They are declared in this class.
 *
 * @sa CPUSolver
 * @sa DynamicParallelism
 * @sa GlobalMemory
 * @sa SharedMemory
 */

#include "Solver.h"

// duplication of this macro
#define CELL(i, j, k, w) ((i) + (j) * (nx + (w)) + (k) * (nx + (w)) * (ny + (w)))

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60


/**
 * Shows progress of the solving process by printing the percentage.
 */
void Solver::printProgress(float percentage)
{
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}


/**
 * Assigns values read by WINDSInputData to variables
 * used in the solvers.
 *
 * @note this is only meant to work with QES-Winds!
 *
 * @param WID :document this:
 * @param WGD :document this:
 */
Solver::Solver(qes::Domain domain_in, const float &tolerance)
  : domain(std::move(domain_in)),
    alpha1(1),
    alpha2(1),
    eta(pow((alpha1 / alpha2), 2.0)),
    A(pow((domain.dx() / domain.dy()), 2.0)),
    B(eta * pow((domain.dx() / domain.dz()), 2.0))

{
  tol = tolerance;

  lambda.resize(domain.numCellCentered(), 0.0);
  lambda_old.resize(domain.numCellCentered(), 0.0);
  R.resize(domain.numCellCentered(), 0.0);
}
