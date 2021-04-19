/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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

/** @file CanopyIsolatedTree.h */

#pragma once

#include <cmath>
#include "util/ParseInterface.h"
#include "CanopyElement.h"

class CanopyIsolatedTree : public CanopyElement
{
public:
  CanopyIsolatedTree()
  {
  }
  CanopyIsolatedTree(const WINDSInputData *WID, WINDSGeneralData *WGD, int id);

  virtual void parseValues()
  {
    parsePrimitive<float>(true, attenuationCoeff, "attenuationCoefficient");
    parsePrimitive<float>(true, H, "height");
    parsePrimitive<float>(true, zMaxLAI, "zMaxLAI");
    parsePrimitive<float>(true, base_height, "baseHeight");
    parsePrimitive<float>(true, x_cent, "xCenter");
    parsePrimitive<float>(true, y_cent, "yCenter");
    // parsePrimitive<float>(true, L, "length");
    parsePrimitive<float>(true, W, "width");
    // parsePrimitive<float>(true, canopy_rotation, "canopyRotation");

    x_start = x_cent - 0.5 * W;
    y_start = y_cent - 0.5 * W;

    L = W;
    canopy_rotation = 0;

    zMaxLAI = zMaxLAI * H;

    // x_start += UID->simParams->halo_x;
    // y_start += UID->simParams->halo_y;
    canopy_rotation *= M_PI / 180.0;
    float f = 36.0;// circle cut in 36 slices
    polygonVertices.resize(f + 1);
    for (int t = 0; t <= f; ++t) {
      polygonVertices[t].x_poly = 0.5 * W * cos(t * 2.0 * M_PI / f) + x_cent;
      polygonVertices[t].y_poly = 0.5 * W * sin(t * 2.0 * M_PI / f) + y_cent;
    }
  }

  void setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_number);
  void canopyVegetation(WINDSGeneralData *wgd, int building_id);
  void canopyWake(WINDSGeneralData *wgd, int building_id);

  int getCellFlagCanopy();
  int getCellFlagWake();

  /*!
   * This function takes in variables read in from input files and initializes required variables for definig
   * canopy elementa.
   */
  // void readCanopy(int nx, int ny, int nz, int landuse_flag, int num_canopies, int &lu_canopy_flag,
  //	std::vector<std::vector<std::vector<float>>> &canopy_atten,std::vector<std::vector<float>> &canopy_top);

private:
  float attenuationCoeff;
  float LAI,zMaxLAI;
  int ustar_method = 2;

  float Bfunc(const float &);
  float ucfunc(const float &, const float &);
};

inline int CanopyIsolatedTree::getCellFlagCanopy()
{
  return 22;
}

inline int CanopyIsolatedTree::getCellFlagWake()
{
  return 23;
}


inline float CanopyIsolatedTree::Bfunc(const float &xh)
{
  if (xh < 7.77) {
    return (0.05 * (xh)) + 2.12;
  } else {
    return 1.70 * pow(xh, 0.19);
  }
}

inline float CanopyIsolatedTree::ucfunc(const float &xh, const float &us)
{
  if (xh < 7.77) {
    // uc=ustar_hanieh*(-0.0935*(xpc/eff_height)*(xpc/eff_height)+10)
    return us * ((-0.63 * (xh)) + 9.33);
  } else {
    return us * 90.68 * pow(xh, -1.48);
  }
}
