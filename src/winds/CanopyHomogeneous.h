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
  CanopyHomogeneous(const std::vector<polyVert> &iSP, float iH, float iBH, float iLAI, int iID);

  virtual void parseValues()
  {
    std::vector<float> xVertex, yVertex;

    x_start = 0;
    y_start = 0;
    L = 0.0;
    W = 0.0;
    building_rotation = 0;

    parsePrimitive<float>(true, attenuationCoeff, "attenuationCoefficient");
    parsePrimitive<float>(true, H, "height");
    parsePrimitive<float>(true, base_height, "baseHeight");

    parsePrimitive<float>(false, x_start, "xStart");
    parsePrimitive<float>(false, y_start, "yStart");
    parsePrimitive<float>(false, L, "length");
    parsePrimitive<float>(false, W, "width");
    parsePrimitive<float>(false, canopy_rotation, "canopyRotation");

    parseMultiPrimitives<float>(false, xVertex, "xVertex");
    parseMultiPrimitives<float>(false, yVertex, "yVertex");

    if (W > 0.0 && L > 0.0) {
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
    } else if (xVertex.size() > 0 && yVertex.size() > 0) {
      int nNodes = xVertex.size();
      polygonVertices.resize(nNodes + 1);
      for (int k = 0; k < nNodes; k++) {
        polygonVertices[k].x_poly = xVertex[k];
        polygonVertices[k].y_poly = yVertex[k];
      }
      polygonVertices[nNodes].x_poly = xVertex[0];
      polygonVertices[nNodes].y_poly = yVertex[0];
    } else {
      std::cerr << "[ERROR] Homogeneous canopy ill-defined" << std::endl;
      exit(1);
    }
  }

  void setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int canopy_id);

  void canopyVegetation(WINDSGeneralData *WGD, int canopy_id);
  void canopyWake(WINDSGeneralData *WGD, int canopy_id)
  {
    return;
  }

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
