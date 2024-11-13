/****************************************************************************
 * Copyright (c) 2024 University of Utah
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
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
 * @file Fire.cpp
 * @brief This clas models fire propagation in the QES framework
 * @version 1.0
 * 
 */
#include "Fire.h"

using namespace std;

Fire ::Fire(WINDSInputData *WID, WINDSGeneralData *WGD)
{
  // get domain information
  nx = WGD->nx;
  ny = WGD->ny;
  nz = WGD->nz;
  dx = WGD->dx;
  dy = WGD->dy;
  dz = WGD->dz;
  FFII_flag = WID->fires->fieldFlag;
  std::cout << "Field flag [" << FFII_flag << "]" << std::endl;
  fmc = WID->fires->fmc;

  /**
	* Set-up the mapper array - cell centered
	**/
  fire_cells.resize((nx - 1) * (ny - 1));
  fuel_map.resize((nx - 1) * (ny - 1));
  burn_flag.resize((nx - 1) * (ny - 1));
  burn_out.resize((nx - 1) * (ny - 1));
  front_map.resize((nx - 1) * (ny - 1));
  del_plus.resize((nx - 1) * (ny - 1));
  del_min.resize((nx - 1) * (ny - 1));
  xNorm.resize((nx - 1) * (ny - 1));
  yNorm.resize((nx - 1) * (ny - 1));
  slope_x.resize((nx - 1) * (ny - 1));
  slope_y.resize((nx - 1) * (ny - 1));
  Force.resize((nx - 1) * (ny - 1));
  z_mix.resize((nx - 1) * (ny - 1));
  z_mix_old.resize((nx - 1) * (ny - 1));
  smoke_flag.resize((nx -1) * (ny - 1));
  /**
	* Read Potential Field
	**/
  // set-up potential field array - cell centered
  Pot_u.resize((nx - 1) * (ny - 1) * (nz - 2));
  Pot_v.resize((nx - 1) * (ny - 1) * (nz - 2));
  Pot_w.resize((nx - 1) * (ny - 1) * (nz - 2));

  // Open netCDF for Potential Field as read only
  NcFile Potential("../data/FireFiles/HeatPot.nc", NcFile::read);

  // Get size of netCDF data
  pot_z = Potential.getVar("u_r").getDim(0).getSize();
  pot_r = Potential.getVar("u_r").getDim(1).getSize();
  pot_G = Potential.getVar("G").getDim(0).getSize();
  pot_rStar = Potential.getVar("rStar").getDim(0).getSize();
  pot_zStar = Potential.getVar("zStar").getDim(0).getSize();
  // Allocate variable arrays
  u_r.resize(pot_z * pot_r);
  u_z.resize(pot_z * pot_r);
  G.resize(pot_G);
  Gprime.resize(pot_G);
  rStar.resize(pot_rStar);
  zStar.resize(pot_zStar);
  // Read start index and length to read
  std::vector<size_t> startIdxField = { 0, 0 };
  std::vector<size_t> countsField = { static_cast<unsigned long>(pot_z),
                                      static_cast<unsigned long>(pot_r) };

  // Get variables from netCDF file
  Potential.getVar("u_r").getVar(startIdxField, countsField, u_r.data());
  Potential.getVar("u_z").getVar(startIdxField, countsField, u_z.data());
  Potential.getVar("G").getVar({ 0 }, { pot_G }, G.data());
  Potential.getVar("Gprime").getVar({ 0 }, { pot_G }, Gprime.data());
  Potential.getVar("rStar").getVar({ 0 }, { pot_rStar }, rStar.data());
  Potential.getVar("zStar").getVar({ 0 }, { pot_zStar }, zStar.data());
  /**
	* Set initial fire info
	*/
  if (FFII_flag == 1) {
    // Open netCDF for fire times
    //std::cout<<"nc file open"<<std::endl;
    NcFile FireTime("../data/FireFiles/FFII.nc", NcFile::read);
    //std::cout<<"nc file read"<<std::endl;
    // Get size of netCDF data
    SFT_time = FireTime.getVar("time").getDim(0).getSize();
    // Allocate variable arrays
    FT_time.resize(SFT_time);
    FT_x1.resize(SFT_time);
    FT_y1.resize(SFT_time);
    FT_x2.resize(SFT_time);
    FT_y2.resize(SFT_time);

    // Get variables from netCDF

    FireTime.getVar("time").getVar({ 0 }, { SFT_time }, FT_time.data());
    FireTime.getVar("y1").getVar({ 0 }, { SFT_time }, FT_x1.data());
    FireTime.getVar("x1").getVar({ 0 }, { SFT_time }, FT_y1.data());
    FireTime.getVar("y2").getVar({ 0 }, { SFT_time }, FT_x2.data());
    FireTime.getVar("x2").getVar({ 0 }, { SFT_time }, FT_y2.data());
    std::cout << "FFII burn times read" << std::endl;
  }
  if (FFII_flag == 2) {
    // Open netCDF for fire times
    //std::cout<<"nc file open"<<std::endl;
    NcFile FireTime("../data/FireFiles/RxCadreL2F.nc", NcFile::read);
    //std::cout<<"nc file read"<<std::endl;
    // Get size of netCDF data
    SFT_time = FireTime.getVar("time").getDim(0).getSize();
    // Allocate variable arrays
    FT_time.resize(SFT_time);
    FT_x1.resize(SFT_time);
    FT_y1.resize(SFT_time);
    FT_x2.resize(SFT_time);
    FT_y2.resize(SFT_time);
    FT_x3.resize(SFT_time);
    FT_y3.resize(SFT_time);

    // Get variables from netCDF

    FireTime.getVar("time").getVar({ 0 }, { SFT_time }, FT_time.data());
    FireTime.getVar("x1").getVar({ 0 }, { SFT_time }, FT_x1.data());
    FireTime.getVar("y1").getVar({ 0 }, { SFT_time }, FT_y1.data());
    FireTime.getVar("x2").getVar({ 0 }, { SFT_time }, FT_x2.data());
    FireTime.getVar("y2").getVar({ 0 }, { SFT_time }, FT_y2.data());
    FireTime.getVar("x3").getVar({ 0 }, { SFT_time }, FT_x3.data());
    FireTime.getVar("y3").getVar({ 0 }, { SFT_time }, FT_y3.data());
    std::cout << "L2F burn times read" << std::endl;
  }

  
  courant = WID->fires->courant;

  for (int fidx = 0; fidx < WID->fires->IG.size(); fidx++) {
    std::cout << "ignition [" << fidx << "]" << std::endl;
    x_start = WID->fires->IG[fidx]->xStart;
    y_start = WID->fires->IG[fidx]->yStart;
    H = WID->fires->IG[fidx]->height;
    L = WID->fires->IG[fidx]->length;
    W = WID->fires->IG[fidx]->width;
    baseHeight = WID->fires->IG[fidx]->baseHeight;
    // get grid info of fire
    i_start = std::round(x_start / dx);
    i_end = std::round((x_start + L) / dx);
    j_start = std::round(y_start / dy);
    j_end = std::round((y_start + W) / dy);
    k_start = std::round((H + baseHeight) / dz);
    k_end = std::round((H + baseHeight) / dz) + 1;

    /**
	* Set-up initial fire state
	*/
    for (int j = j_start; j < j_end; j++) {
      for (int i = i_start; i < i_end; i++) {
        int idx = i + j * (nx - 1);
        fire_cells[idx].state.burn_flag = 1;
        fire_cells[idx].state.front_flag = 1;
	
      }
    }
  }
  /** 
	*  Set up burn flag field and smoke flag field
	*/
  for (int j = 0; j < ny - 1; j++) {
    for (int i = 0; i < nx - 1; i++) {
      int idx = i + j * (nx - 1);
      burn_flag[idx] = fire_cells[idx].state.burn_flag;
      smoke_flag[idx] = fire_cells[idx].state.burn_flag;
    }
  }
  std::cout << "burn initialized" << std::endl;
  /**
	* Set up initial level set. Use signed distance function: swap to fast marching method in future.
	*/
  float sdf, sdf_min;
  for (int j = 0; j < ny - 1; j++) {
    for (int i = 0; i < nx - 1; i++) {
      int idx = i + j * (nx - 1);
      if (fire_cells[idx].state.front_flag == 1) {
        front_map[idx] = 0;
      } else {
        sdf = 1000;
        for (int jj = 0; jj < ny - 1; jj++) {
          for (int ii = 0; ii < nx - 1; ii++) {
            int idx2 = ii + jj * (nx - 1);
            if (fire_cells[idx2].state.front_flag == 1) {
              sdf_min = sqrt((ii - i) * (ii - i) + (jj - j) * (jj - j));
            } else {
              sdf_min = 1000;
            }
            sdf = sdf_min < sdf ? sdf_min : sdf;
          }
        }
        front_map[idx] = sdf;
      }
    }
  }
  std::cout << "level set initialized" << std::endl;
  /**
	* Calculate slope at each terrain cell
	*/
  for (int j = 1; j < ny - 2; j++) {
    for (int i = 1; i < nx - 2; i++) {
      int id = i + j * (nx - 1);
      int idxp = (i + 1) + j * (nx - 1);
      int idxm = (i - 1) + j * (nx - 1);
      int idyp = i + (j + 1) * (nx - 1);
      int idym = i + (j - 1) * (nx - 1);
      float delzx = WGD->terrain[idxp] - WGD->terrain[idxm];
      float delzy = WGD->terrain[idyp] - WGD->terrain[idym];
      slope_x[id] = (delzx / (2 * dx)) / sqrt(delzx * delzx + 2 * dx * 2 * dx);
      slope_y[id] = (delzy / (2 * dy)) / sqrt(delzy * delzy + 2 * dy * 2 * dy);
    }
  }
  std::cout << "slope calculated" << std::endl;
}

/**
 * Compute adaptive time step. Based on Courant criteria.
 */
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

/**
 * Compute fire spread for burning cells
 */
void Fire ::run(Solver *solver, WINDSGeneralData *WGD)
{


  /**
     * Calculate level set gradient and norm (Chapter 6, Sethian 2008)
     */
  float dmx, dpx, dmy, dpy, n_star_x, n_star_y;
  float sdmx, sdpx, sdmy, sdpy, sn_star_x, sn_star_y;
  for (int j = 1; j < ny - 2; j++) {
    for (int i = 1; i < nx - 2; i++) {
      int idx = i + j * (nx - 1);
      int idxjp = i + (j + 1) * (nx - 1);
      int idxjm = i + (j - 1) * (nx - 1);
      dmy = (front_map[idx] - front_map[idxjm]) / dx;
      dpy = (front_map[idxjp] - front_map[idx]) / dx;
      dmx = (front_map[idx] - front_map[idx - 1]) / dy;
      dpx = (front_map[idx + 1] - front_map[idx]) / dy;
      del_plus[idx] = sqrt(fmax(dmx, 0) * fmax(dmx, 0) + fmin(dpx, 0) * fmin(dpx, 0) + fmax(dmy, 0) * fmax(dmy, 0) + fmin(dpy, 0) * fmin(dpy, 0));
      del_min[idx] = sqrt(fmax(dpx, 0) * fmax(dpx, 0) + fmin(dmx, 0) * fmin(dmx, 0) + fmax(dpy, 0) * fmax(dpy, 0) + fmin(dmy, 0) * fmin(dmy, 0));
      n_star_x = dpx / sqrt(dpx * dpx + dpy * dpy) + dmx / sqrt(dmx * dmx + dpy * dpy) + dpx / sqrt(dpx * dpx + dmy * dmy) + dmx / sqrt(dmx * dmx + dmy * dmy);
      n_star_y = dpy / sqrt(dpx * dpx + dpy * dpy) + dpy / sqrt(dmx * dmx + dpy * dpy) + dmy / sqrt(dpx * dpx + dmy * dmy) + dmy / sqrt(dmx * dmx + dmy * dmy);
      xNorm[idx] = n_star_x / sqrt(n_star_x * n_star_x + n_star_y * n_star_y);
      yNorm[idx] = n_star_y / sqrt(n_star_x * n_star_x + n_star_y * n_star_y);
    }
  }
  //std::cout << "level set calculated" << std::endl;
  /**
	* Reset forcing function for level set
	*/
  std::fill(Force.begin(), Force.end(), 0);
  /**
  float terrain_min=WGD->terrain[0];
  for (int j = 1; j < ny-2; j++) {
    for (int i = 1; i< nx-2; i++) {
      int idx = i + j*(nx-1);
      float ter_min = WGD->terrain[idx];
      if (ter_min < terrain_min) {
	terrain_min = ter_min;
      }
    }
  }
  std::cout<<"Minimum terrain = "<<terrain_min<<std::endl;
  */
  /**
	* Calculate Forcing Function (Balbi model at mid-flame height or first grid cell if no fire)
	*/

  for (int j = 1; j < ny - 2; j++) {
    for (int i = 1; i < nx - 2; i++) {
      int idx = i + j * (nx - 1);
      // get fuel properties at this location
      struct FuelProperties *fuel = fire_cells[idx].fuel;
      // calculate mid-flame height
      int kh = 0;
      float H = fire_cells[idx].properties.h;
      float T = WGD->terrain[idx];
      float D = fuel->fuelDepth * 0.3048;
      float FD = H + T + D;

      if (H == 0) {
        kh = std::round(T / dz);
      } else {
        kh = std::round(FD / dz);
      }
      // call u and v from WINDS General Data
      int cell_face = i + j * nx + kh * nx * ny;
      float u = 0.5 * (WGD->u[cell_face] + WGD->u[cell_face + 1]);
      float v = 0.5 * (WGD->v[cell_face] + WGD->v[cell_face + nx]);
      // run Balbi model
      struct FireProperties fp = balbi(fuel, u, v, xNorm[idx], yNorm[idx], slope_x[idx], slope_y[idx], fmc);
      fire_cells[idx].properties = fp;
      Force[idx] = fp.r;
    }
  }
  //std::cout << "Level set forcing calculated" << std::endl;
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
    int idx = ii + jj * (nx-1);
    float T = WGD->terrain[idx];
    float D = fuel->fuelDepth * 0.3048;
    int TID = std::round(T / dz);
    float FD = H / 2.0 + T + D;
    float MFD = maxH + T + D;
    if (H == 0) {
      kh = std::round(T / dz);
    } else {
      kh = std::round(FD / dz);
    }
    if (maxH == 0) {
      maxkh = std::round(T / dz);
    } else {
      maxkh = std::round(MFD / dz);
    }

    // get horizontal wind at flame height
    int cell_face = ii + jj * nx + (kh) * ny * nx;
    float u = 0.5 * (WGD->u[cell_face] + WGD->u[cell_face + 1]);
    float v = 0.5 * (WGD->v[cell_face] + WGD->v[cell_face + nx]);
    // run Balbi model
    float burnTime = fire_cells[id].state.burn_time;
    struct FireProperties fp = balbi(fuel, u, v, xNorm[id], yNorm[id], slope_x[id], slope_y[id], fmc);
    fire_cells[id].properties = fp;
    Force[id] = fp.r;
    // update icell value for flame
    for (int k = TID; k <= maxkh; k++) {
      int icell_cent = ii + jj * (nx - 1) + (k) * (nx - 1) * (ny - 1);
      WGD->icellflag[icell_cent] = 12;
      // std::cout<<"TID = "<<TID<<", T = "<<T<<", i = "<<ii<<", j = "<<jj<<std::endl;
    }
  }
  // compute time step
  dt = computeTimeStep();
  std::vector<int>().swap(cells_burning);
}

/** 
 * Compute fire spread. Advance level set.
 */
void Fire ::move(Solver *solver, WINDSGeneralData *WGD)
{

  if (FFII_flag == 1) {
    int FT_idx1 = 0;
    int FT_idx2 = 0;
    float FT = ceil(time) + FT_time[0];
    int it;
    for (int IDX = 0; IDX < FT_time.size(); IDX++) {
      if (FT == FT_time[IDX]) {
        it = IDX;
        break;
      }
    }
    int nx1 = round(FT_x1[it] / dx);
    int ny1 = round((750 - FT_y1[it]) / dy);
    FT_idx1 = nx1 + ny1 * (nx - 1);
    int nx2 = round(FT_x2[it] / dx);
    int ny2 = round((750 - FT_y2[it]) / dy);
    FT_idx2 = nx2 + ny2 * (nx - 1);
    if (burn_flag[FT_idx1] < 2) {
      front_map[FT_idx1] = 0;
      fire_cells[FT_idx1].state.burn_flag = 1;
    }
    if (burn_flag[FT_idx2] < 2) {
      front_map[FT_idx2] = 0;
      fire_cells[FT_idx2].state.burn_flag = 1;
    }
  }
  if (FFII_flag == 2) {
    int FT_idx1 = 0;
    int FT_idx2 = 0;
    int FT_idx3 = 0;
    float FT = ceil(time) + FT_time[0];
    int it = -1;
    for (int IDX = 0; IDX < FT_time.size(); IDX++) {
      if (FT == FT_time[IDX]) {
        it = IDX;
        break;
      }
    }
    if (it >= 0) {
      int nx1 = round(FT_x1[it] / dx);
      int ny1 = round((FT_y1[it]) / dy);
      FT_idx1 = nx1 + ny1 * (nx - 1);
      int nx2 = round(FT_x2[it] / dx);
      int ny2 = round((FT_y2[it]) / dy);
      FT_idx2 = nx2 + ny2 * (nx - 1);
      int nx3 = round(FT_x3[it] / dx);
      int ny3 = round((FT_y3[it]) / dy);
      FT_idx3 = nx3 + ny3 * (nx - 1);
      if (burn_flag[FT_idx1] < 2) {
        front_map[FT_idx1] = 0;
        fire_cells[FT_idx1].state.burn_flag = 1;
      }
      if (burn_flag[FT_idx2] < 2) {
        front_map[FT_idx2] = 0;
        fire_cells[FT_idx2].state.burn_flag = 1;
      }
      if (burn_flag[FT_idx3] < 2) {
        front_map[FT_idx3] = 0;
        fire_cells[FT_idx3].state.burn_flag = 1;
      }
    }
  }
  for (int j = 1; j < ny - 2; j++) {
    for (int i = 1; i < nx - 2; i++) {
      int idx = i + j * (nx - 1);
      // get fire properties at this location
      struct FireProperties fp = fire_cells[idx].properties;
      struct FuelProperties *fuel = fire_cells[idx].fuel;
      float H = fire_cells[idx].properties.h * (1 - (fire_cells[idx].state.burn_time / fire_cells[idx].properties.tau));
      float maxH = fire_cells[idx].properties.h;
      float T = WGD->terrain[idx];
      float D = fuel->fuelDepth * 0.3048;
      int TID = std::round(T / dz);
      float FD = H / 2.0 + T + D;
      float MFD = maxH + T + D;
      int kh = 0;
      int maxkh = 0;

      if (H == 0) {
        kh = std::round(T / dz);
      } else {
        kh = std::round(FD / dz);
      }
      if (maxH == 0) {
        maxkh = std::round(T / dz);
      } else {
        maxkh = std::round(MFD / dz);
      }

      // if burn flag = 1, update burn time
      if (burn_flag[idx] == 1) {
        fire_cells[idx].state.burn_time += dt;
      }
      // set burn flag to 2 (burned) if residence time exceeded, set Forcing function to 0, and update z0 to bare soil
      if (fire_cells[idx].state.burn_time >= fp.tau) {
        fire_cells[idx].state.burn_flag = 2;
        Force[idx] = 0;

        // Need to fix where z0 is reset MM
        //WGD->z0_domain[idx] = 0.01;
      }

      // advance level set

      front_map[idx] = front_map[idx] - dt * (fmax(Force[idx], 0) * del_plus[idx] + fmin(Force[idx], 0) * del_min[idx]);
      // if level set <= 1, set burn_flag to 0.5 - L.S. for preheating
      if (front_map[idx] <= 1 && burn_flag[idx] < 1) {
        fire_cells[idx].state.burn_flag = 0.5;
      }
      // if level set < threshold, set burn flag to 1 and start smoke flag
      if (front_map[idx] <= 0.1 && burn_flag[idx] < 1) {
        fire_cells[idx].state.burn_flag = 1;
	smoke_flag[idx] = 1;
      }
      // update burn flag field
      burn_flag[idx] = fire_cells[idx].state.burn_flag;
      burn_out[idx] = burn_flag[idx];
    }
  }
  // advance time
  time += dt;
}