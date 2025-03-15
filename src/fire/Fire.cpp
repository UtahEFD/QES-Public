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
 * @file Fire.cpp
 * @brief This class models fire propagation in the QES framework
 */
#include "Fire.h"

using namespace std;

Fire::Fire(WINDSInputData *WID, WINDSGeneralData *WGD)
{
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "[QES-Fire]\t Initialization of fire model...\n";

  // get domain information
  nx = WGD->domain.nx();
  ny = WGD->domain.ny();
  nz = WGD->domain.nz();
  dx = WGD->domain.dx();
  dy = WGD->domain.dy();
  dz = WGD->domain.dz();
  FFII_flag = 0;
  fmc = WID->fires->fmc;
  cure = WID->fires->cure;
  if (cure < .30) {
    cure = .30;
  } else if (cure > 1.20) {
    cure = 1.20;
  }
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
  smoke_flag.resize((nx - 1) * (ny - 1));
  H0.resize((nx - 1) * (ny - 1));
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
  Potential.getVar("G").getVar({ 0 }, { static_cast<unsigned long>(pot_G) }, G.data());
  Potential.getVar("Gprime").getVar({ 0 }, { static_cast<unsigned long>(pot_G) }, Gprime.data());
  Potential.getVar("rStar").getVar({ 0 }, { static_cast<unsigned long>(pot_rStar) }, rStar.data());
  Potential.getVar("zStar").getVar({ 0 }, { static_cast<unsigned long>(pot_zStar) }, zStar.data());
  /**
   * Set initial fire info
   */
  std::string igFile = WID->fires->igFile;
  if (igFile != "") {
    FFII_flag = 1;
    // Open netCDF for fire times
    NcFile FireTime(igFile, NcFile::read);
    // Get size of netCDF data
    SFT_time = FireTime.getVar("time").getDim(0).getSize();
    // Allocate variable arrays
    FT_time.resize(SFT_time);
    FT_x1.resize(SFT_time);
    FT_y1.resize(SFT_time);

    // Get variables from netCDF
    FireTime.getVar("time").getVar({ 0 }, { static_cast<unsigned long>(SFT_time) }, FT_time.data());
    FireTime.getVar("y1").getVar({ 0 }, { static_cast<unsigned long>(SFT_time) }, FT_x1.data());
    FireTime.getVar("x1").getVar({ 0 }, { static_cast<unsigned long>(SFT_time) }, FT_y1.data());

    std::cout << "[QES-Fire]\t Ignition file " << igFile << " read succesfully" << std::endl;
  }

  courant = WID->fires->courant;

  for (int fidx = 0; fidx < WID->fires->IG.size(); fidx++) {
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
   * Set up burn flag field and smoke flag field
   */
  for (int j = 0; j < ny - 1; j++) {
    for (int i = 0; i < nx - 1; i++) {
      int idx = i + j * (nx - 1);
      burn_flag[idx] = fire_cells[idx].state.burn_flag;
      smoke_flag[idx] = fire_cells[idx].state.burn_flag;
    }
  }
  std::cout << "[QES-Fire]\t Burn initialized" << std::endl;

#ifdef HAS_CUDA
  if (potFlag == 1) {
    LSinitGlob();
  } else {
    LSinit();
  }
#else
  LSinit();
#endif
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
  std::cout << "[QES-Fire]\t Slope Calculated" << std::endl;
}
