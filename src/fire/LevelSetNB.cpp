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
 * @file LevelSet.cpp
 * @brief This function calculates the fire progression by advancement of a level set
 */
#include "Fire.h"

void Fire ::LevelSetNB(WINDSGeneralData *WGD)
{
  /**
   * Reset forcing function for level set
   */
  std::fill(Force.begin(), Force.end(), 0);
 
  // indices for burning cells
  std::vector<int> cells_burning;
  // search predicate for burn state
  struct find_burn : std::unary_function<FireCell, bool>
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
    // get fuel properties at this location
    struct FuelProperties *fuel = fire_cells[id].fuel;
    // get vertical index of mid-flame height
    int kh = 0;///< mid-flame height
    int maxkh = 0;///< max flame height
    //modify flame height by time on fire (assume linear functionality)
    float H = fire_cells[id].properties.h * (1 - (fire_cells[id].state.burn_time / fire_cells[id].properties.tau));
    float maxH = fire_cells[id].properties.h;

    // convert flat index to i, j at cell center
    int ii = id % (nx - 1);
    int jj = (id / (nx - 1)) % (ny - 1);
    int IDX = ii + jj * (nx-1);
    float T = WGD->terrain[IDX];
    float D = fuel->fuelDepth * 0.3048;
    int TID = std::round(T / dz);
    float FD = H / 2.0 + T + D;
    float MFD = maxH + T + D;
    if (H <= 0) {
      kh = std::round(T / dz);
    } else {
      kh = std::round(FD / dz);
    }
    if (maxH == 0) {
      maxkh = std::round(T / dz);
    } else {
      maxkh = std::round(MFD / dz);
    }
    kh = kh > 0 ? kh : 0;
    
    /**
     * Calculate level set gradient and norm in narrow band (+/-2 cells)(Chapter 6,7, Sethian 2008)
     */
    int xmin = ii-2 > 1 ? ii-2 : 1;
    int xmax = ii+2 < nx-3 ? ii+2 : nx-3;
    int ymin = jj-2 > 1 ? jj-2 : 1;
    int ymax = jj+2 < ny-3 ? jj+2 : ny-3;
    
    float dmx, dpx, dmy, dpy, n_star_x, n_star_y;
    float sdmx, sdpx, sdmy, sdpy, sn_star_x, sn_star_y;
    for (int n = ymin; n <= ymax; n++) {
      for (int m = xmin; m <= xmax; m++) {
	//std::cout<<"[m][n]:["<<m<<"]["<<n<<"]"<<std::endl;
	int idx = m + n * (nx - 1);
	int idxjp = m + (n + 1) * (nx - 1);
	int idxjm = m + (n - 1) * (nx - 1);

	dmy = (front_map[idx] - front_map[idxjm]) / dx;
	dpy = (front_map[idxjp] - front_map[idx]) / dx;
	dmx = (front_map[idx] - front_map[idx - 1]) / dy;
	dpx = (front_map[idx + 1] - front_map[idx]) / dy;

	del_plus[idx] = sqrt(fmax(dmx, 0) * fmax(dmx, 0) + fmin(dpx, 0) * fmin(dpx, 0) + fmax(dmy, 0) * fmax(dmy, 0) + fmin(dpy, 0) * fmin(dpy, 0));
	del_min[idx] = sqrt(fmax(dpx, 0) * fmax(dpx, 0) + fmin(dmx, 0) * fmin(dmx, 0) + fmax(dpy, 0) * fmax(dpy, 0) + fmin(dmy, 0) * fmin(dmy, 0));
	n_star_x = dpx / sqrt(dpx * dpx + dpy * dpy) + dmx / sqrt(dmx * dmx + dpy * dpy) + dpx / sqrt(dpx * dpx + dmy * dmy) + dmx / sqrt(dmx * dmx + dmy * dmy);
	n_star_y = dpy / sqrt(dpx * dpx + dpy * dpy) + dpy / sqrt(dmx * dmx + dpy * dpy) + dmy / sqrt(dpx * dpx + dmy * dmy) + dmy / sqrt(dmx * dmx + dmy * dmy);
	if (n_star_x == 0){
	  xNorm[idx] = 0;
	} else {
	  xNorm[idx] = n_star_x / sqrt(n_star_x * n_star_x + n_star_y * n_star_y);
	}
	if (n_star_y == 0){
	  yNorm[idx] = 0;
	} else {
	  yNorm[idx] = n_star_y / sqrt(n_star_x * n_star_x + n_star_y * n_star_y);
	}


	// get horizontal wind at flame height
	int cell_face = m + n * nx + (kh) * ny * nx;
	float u = 0.5 * (WGD->u[cell_face] + WGD->u[cell_face + 1]);
	float v = 0.5 * (WGD->v[cell_face] + WGD->v[cell_face + nx]);
	// run Balbi model
	//float burnTime = fire_cells[id].state.burn_time;
	struct FireProperties fp = balbi(fuel, u, v, xNorm[idx], yNorm[idx], slope_x[idx], slope_y[idx], fmc);
	fire_cells[idx].properties = fp;
	Force[idx] = fp.r;
	
      }
    }
    // update icell value for flame
	for (int k = TID; k <= maxkh; k++) {
	  int icell_cent = ii + jj * (nx - 1) + (k) * (nx - 1) * (ny - 1);
	  WGD->icellflag[icell_cent] = 12;
	}
  }

  // compute time step
  dt = computeTimeStep();

  std::vector<int>().swap(cells_burning);
}
