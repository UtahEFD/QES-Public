/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file PotentialGlobal.h
 */
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

#include "../Fire.h"

template<typename T>
  void _cudaCheck(T e, const char *func, const char *call, const int line);

// potential velocity fields on device
float *d_Pot_u, *d_Pot_v, *d_Pot_w;

// potential variables
float *d_u_r, *d_u_z, *d_G, *d_Gprime;

// initial velocity fields on device
float *d_u0, *d_v0, *d_w0;

int *d_icellflag;