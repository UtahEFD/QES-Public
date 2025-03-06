/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Matthew Moody
 * Copyright (c) 2025 Jeremy Gibbs
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Brian Bailey
 * Copyright (c) 2025 Pete Willemsen
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
  
  // indices for burning cells
  std::vector<int> cells_burning;
  // search predicate for burn state
  struct find_burn //: std::__unary_function<FireCell, bool>
  {
    float burn;
    find_burn(int burn) : burn(burn) {}
    bool operator()(FireCell const &f) const
    {
      return f.state.burn_flag == burn;
    }
  };

  // get indices of burning cells
  std::vector<FireCell>::iterator it = std::find_if(fire_cells.begin(), fire_cells.end(), find_burn(1));
  while (it != fire_cells.end()) {
    if (it != fire_cells.end()) {
      cells_burning.push_back(std::distance(fire_cells.begin(), it));
    }
    it = std::find_if(++it, fire_cells.end(), find_burn(1));
  }

  
  // loop through burning cells
  for (int i = 0; i < cells_burning.size(); i++) {
    // get index burning cell
    int id = cells_burning[i];
    // convert flat index to i, j at cell center
    int ii = id % (nx - 1);
    int jj = (id / (nx - 1)) % (ny - 1);
    int idx = ii + jj * (nx-1);
    
    r = fire_cells[idx].properties.r;
    r_max = r > r_max ? r : r_max;
  }
  
  std::cout << "max ROS = " << r_max << "[ms^-1]" << std::endl;
  
  if (r_max < 0.05) {
    r_max = 0.05;
  }
  else if (isnan(r_max)){
    r_max = 0.05;
    std::cout<<"r_max is NaN, setting to 0.3"<<std::endl;
  }
  
  float dt = courant * dx / r_max;
 
  std::cout << "dt = " << dt << " s" << std::endl;
  return courant * dx / r_max;
}
