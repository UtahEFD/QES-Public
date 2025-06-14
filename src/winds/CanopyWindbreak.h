/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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

/** @file CanopyWindbreak.h */

#pragma once

#include <cmath>
#include "util/ParseInterface.h"
#include "CanopyElement.h"

class CanopyWindbreak : public CanopyElement
{
public:
  CanopyWindbreak()
  {
  }


  virtual void parseValues()
  {
    base_height = 0.0;

    parsePrimitive<float>(true, H, "height");
    parsePrimitive<float>(false, base_height, "baseHeight");
    parsePrimitive<float>(true, understory_height, "understroyHeight");
    parsePrimitive<float>(true, x_start, "xStart");
    parsePrimitive<float>(true, y_start, "yStart");
    parsePrimitive<float>(true, L, "length");
    parsePrimitive<float>(true, W, "width");
    parsePrimitive<float>(true, canopy_rotation, "canopyRotation");

    parsePrimitive<float>(true, beta, "opticalPorosity");

    parsePrimitive<int>(false, wbModel, "fenceModel");
    parsePrimitive<float>(false, fetch, "fetch");

    // x_start += UID->simParams->halo_x;
    // y_start += UID->simParams->halo_y;
    double pi = 3.14159265359;
    canopy_rotation *= pi / 180.0;
    polygonVertices.resize(5);
    polygonVertices[0].x_poly = polygonVertices[4].x_poly = x_start;
    polygonVertices[0].y_poly = polygonVertices[4].y_poly = y_start;
    polygonVertices[1].x_poly = x_start - L * sin(canopy_rotation);
    polygonVertices[1].y_poly = y_start + L * cos(canopy_rotation);
    polygonVertices[2].x_poly = polygonVertices[1].x_poly + W * cos(canopy_rotation);
    polygonVertices[2].y_poly = polygonVertices[1].y_poly + W * sin(canopy_rotation);
    polygonVertices[3].x_poly = x_start + W * cos(canopy_rotation);
    polygonVertices[3].y_poly = y_start + W * sin(canopy_rotation);
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
  // float attenuationCoeff=1.0;
  float understory_height = 0.0;

  float beta = 0.01;// optical porosity
  int wbModel = 2;// flow in windbreak 1 for wang aerodynamic profile; bean otherwise
  float a_obf;// bleed flow areo porosity
  float d = 0.0;// from upwind profile
  float stdw = 0.19;// upstream vertical variance

  float fetch = 7;

  // std::map<int, float> u0, v0;
};

inline int CanopyWindbreak::getCellFlagCanopy()
{
  return 24;
}

inline int CanopyWindbreak::getCellFlagWake()
{
  return 25;
}
