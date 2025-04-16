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

/** @file SolverFactory.h */

#pragma once

#include "Solver.h"
#include "Solver_CPU.h"
#include "Solver_CPU_RB.h"
#ifdef HAS_CUDA
// While we get this verified on CUDA 12.8, we will
// replace use of it with the GlobalMemory solver.
// #include "winds/Solver_GPU_DynamicParallelism.h"
#include "Solver_GPU_GlobalMemory.h"
#include "Solver_GPU_SharedMemory.h"
#endif

#include "../qes/Domain.h"

enum solverTypes : int { CPU_Type = 1,
                         DYNAMIC_P = 2,
                         Global_M = 3,
                         Shared_M = 4 };


class SolverFactory
{
public:
  static Solver *create(const int &, const qes::Domain &, const float &);
};