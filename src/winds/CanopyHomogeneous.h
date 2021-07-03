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

/** @file CanopyHomogeneous.h */

#pragma once

#include <cmath>
#include "util/ParseInterface.h"
#include "CanopyElement.h"

class CanopyHomogeneous : public CanopyElement
{
public:
  CanopyHomogeneous()
  {
  }


  virtual void parseValues()
  {
    parsePrimitive<float>(true, attenuationCoeff, "attenuationCoefficient");
    parsePrimitive<float>(true, H, "height");
    parsePrimitive<float>(true, base_height, "baseHeight");
    parsePrimitive<float>(true, x_start, "xStart");
    parsePrimitive<float>(true, y_start, "yStart");
    parsePrimitive<float>(true, L, "length");
    parsePrimitive<float>(true, W, "width");
    parsePrimitive<float>(true, canopy_rotation, "canopyRotation");


    // x_start += UID->simParams->halo_x;
    // y_start += UID->simParams->halo_y;
    canopy_rotation *= M_PI / 180.0;
    polygonVertices.resize(5);
    polygonVertices[0].x_poly = polygonVertices[4].x_poly = x_start;
    polygonVertices[0].y_poly = polygonVertices[4].y_poly = y_start;
    polygonVertices[1].x_poly = x_start - W * sin(canopy_rotation);
    polygonVertices[1].y_poly = y_start + W * cos(canopy_rotation);
    polygonVertices[2].x_poly = polygonVertices[1].x_poly + L * cos(canopy_rotation);
    polygonVertices[2].y_poly = polygonVertices[1].y_poly + L * sin(canopy_rotation);
    polygonVertices[3].x_poly = x_start + L * cos(canopy_rotation);
    polygonVertices[3].y_poly = y_start + L * sin(canopy_rotation);
  }

  void setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_id);

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
};

inline int CanopyHomogeneous::getCellFlagCanopy()
{
  return 20;
}

inline int CanopyHomogeneous::getCellFlagWake()
{
  return 21;
}
