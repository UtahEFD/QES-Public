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
 * @file Fire.h
 * @brief This class models fire propagation in the QES framework
 */
#ifndef FIRE_H
#define FIRE_H

#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"
#include "FuelProperties.hpp"
#include "util/Vector3.h"
#include "util/Vector3Int.h"
#include "winds/Solver.h"
#include "FuelRead.h"
#include "winds/DTEHeightField.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <netcdf>
#include "SourceFire.h"
#include "Smoke.h"

using namespace netCDF;
using namespace netCDF::exceptions;
using namespace std::chrono;
using namespace std;

class FuelProperties;
//class Smoke;

class Fire
{


public:

  Fire(WINDSInputData *, WINDSGeneralData *);


  struct FireProperties
  {
    float w, h, d, r, T, tau, K, H0, U_c, L_c;
  };

  struct FireState
  {
    float burn_time;
    float burn_flag;
    float front_flag;
  };

  struct FireCell
  {
    FireProperties properties;
    FuelProperties *fuel;
    FireState state;
  };

  float PI = 3.14159265359;
  double time = 0;
  float r_max = 0;
  float dt = 0;
  int FFII_flag = 0;

  std::vector<FireCell> fire_cells;

  std::vector<float> w_base;

  std::vector<float> front_map;

  std::vector<float> del_plus;

  std::vector<float> del_min;

  std::vector<float> xNorm;

  std::vector<float> yNorm;
  std::vector<float> slope_x;
  std::vector<float> slope_y;
  std::vector<float> Force;
  std::vector<float> z_mix;
  std::vector<float> z_mix_old;
  std::vector<float> Pot_u;
  std::vector<float> Pot_v;
  std::vector<float> Pot_w;


  // output fields
  std::vector<float> burn_flag;
  std::vector<float> burn_out;
  std::vector<int> smoke_flag;
  std::vector<float> Pot_w_out;
  std::vector<float> fuel_map;


  // Potential field
  int pot_z, pot_r, pot_G, pot_rStar, pot_zStar;
  float drStar, dzStar;
  std::vector<float> u_r;
  std::vector<float> u_z;
  std::vector<float> G;
  std::vector<float> Gprime;
  std::vector<float> rStar;
  std::vector<float> zStar;

  // Fire Arrival Times from netCDF
  int SFT_time, SFT_x1, SFT_y1, SFT_x2, SFT_y2;
  std::vector<float> FT_time;
  std::vector<float> FT_x1;
  std::vector<float> FT_y1;
  std::vector<float> FT_x2;
  std::vector<float> FT_y2;
  std::vector<float> FT_x3;
  std::vector<float> FT_y3;


  void LevelSet(WINDSGeneralData *);
  void LevelSetNB(WINDSGeneralData *);
  void move(WINDSGeneralData *);
  void potential(WINDSGeneralData *);
  void FuelMap(WINDSInputData *, WINDSGeneralData *);
  float computeTimeStep();

private:
  // grid information
  int nx, ny, nz;
  float dx, dy, dz;

  // fire information
  int fuel_type;
  float fmc;
  float cure;
  float x_start, y_start;
  float L, W, H, baseHeight, courant;
  int i_start, i_end, j_start, j_end, k_end, k_start;
  int fieldFlag;
  float rothermel(FuelProperties *, float, float, float);

  FireProperties balbi(FuelProperties *, float, float, float, float, float, float, float);
  

  FireProperties runFire(float, float, int);
};

#endif
