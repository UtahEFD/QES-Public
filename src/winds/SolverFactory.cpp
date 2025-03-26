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

/** @file SolverFactory.cpp */

#include "SolverFactory.h"

Solver *SolverFactory::create(const int &solveType, const qes::Domain &domain, const float &tolerance)
{
  Solver *solver = nullptr;
  if (solveType == CPU_Type) {
#ifdef _OPENMP
    solver = new Solver_CPU_RB(domain, tolerance);
#else
    solver = new Solver_CPU(domain, tolerance);
#endif

#ifdef HAS_CUDA
  } else if (solveType == DYNAMIC_P) {
    // While we get this verified on CUDA 12.8, we will
    // replace use of it with the GlobalMemory solver.
    // solver = new Solver_GPU_DynamicParallelism(domain, tolerance);
    std::cout << "The Global Memory GPU solver will be used in place of the Dynamic Parallelism GPU Solver for the time being." << std::endl;
    solver = new Solver_GPU_GlobalMemory(domain, tolerance);
  } else if (solveType == Global_M) {
    solver = new Solver_GPU_GlobalMemory(domain, tolerance);
  } else if (solveType == Shared_M) {
    solver = new Solver_GPU_SharedMemory(domain, tolerance);
#endif
  } else {
    QESout::error("Invalid solver type");
  }
  return solver;
}