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
 * @file ComputeTimeStep.cpp
 * @brief This function computes the dynamic timestep for fire grid based on Courant number
 */
#include "Fire.h"

float Fire ::computeTimeStep()
{
  // spread rates
  float r = 0;
  float r_max = 0;

  // get max spread rate
  for (int j = 0; j < ny - 1; j++) {
    for (int i = 0; i < nx - 1; i++) {
      int idx = i + j * (nx - 1);
      r = fire_cells[idx].properties.r;
      r_max = r > r_max ? r : r_max;
    }
  }
  std::cout << "max ROS = " << r_max << std::endl;
  if (r_max < 0.3) {
    r_max = 0.3;
  }
  else if (isnan(r_max)){
    r_max = 0.3;
    std::cout<<"r_max is NaN, setting to 0.3"<<std::endl;
  }
  float dt = courant * dx / r_max;
 
  std::cout << "dt = " << dt << " s" << std::endl;
  return courant * dx / r_max;
}