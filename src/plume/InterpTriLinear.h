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
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file InterpTriLinear.h
 * @brief
 */

#pragma once

#include <iostream>
#include <ctime>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "util/calcTime.h"
#include "Random.h"
#include "util/Vector3Float.h"

#include "Interp.h"

struct interpWeight
{
  int ii;// nearest cell index to the left in the x direction
  int jj;// nearest cell index to the left in the y direction
  int kk;// nearest cell index to the left in the z direction
  float iw;// normalized distance to the nearest cell index to the left in the x direction
  float jw;// normalized distance to the nearest cell index to the left in the y direction
  float kw;// normalized distance to the nearest cell index to the left in the z direction
};

class InterpTriLinear : public Interp
{

public:
  // constructor
  // copies the turb grid values for nx, ny, nz, nt, dx, dy, and dz to the InterpTriLinear grid values,
  // then calculates the tau gradients which are then used to calculate the flux_div grid values.
  InterpTriLinear(qes::Domain, bool);

  // double vel_threshold;

  void interpWindsValues(const WINDSGeneralData *WGD,
                         const vec3 &pos,
                         vec3 &vel_out) override;

  void interpTurbValues(const TURBGeneralData *TGD,
                        const vec3 &pos,
                        mat3sym &tau_out,
                        vec3 &flux_div_out,
                        float &nuT_out,
                        float &CoEps_out) override;

  void interpTurbInitialValues(const TURBGeneralData *TGD,
                               const vec3 &pos,
                               mat3sym &tau_out,
                               vec3 &sig_out) override;

private:
  void setInterp3Dindex_uFace(const vec3 &, interpWeight &);
  void setInterp3Dindex_vFace(const vec3 &, interpWeight &);
  void setInterp3Dindex_wFace(const vec3 &, interpWeight &);
  void interp3D_faceVar(const std::vector<float> &, const interpWeight &, float &);
  void interp3D_faceVar(const std::vector<double> &, const interpWeight &, double &);

  void setInterp3Dindex_cellVar(const vec3 &, interpWeight &);
  void interp3D_cellVar(const std::vector<float> &, const interpWeight &, float &);
  void interp3D_cellVar(const std::vector<double> &, const interpWeight &, double &);

  // copies of debug related information from the input arguments
  bool debug{};
};
