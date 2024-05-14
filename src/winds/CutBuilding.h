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

/** @file CutBuilding.h */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <algorithm>

#include "util/ParseInterface.h"
#include "Building.h"


using namespace std;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;

#define MIN_S(x, y) ((x) < (y) ? (x) : (y))

/**
 * @class CutBuliding
 * @brief Designed for applying the cut-cell method to the general building shape (polygons).
 *
 * It's an inheritance of the Building class (has all the features defined in that class).
 * In this class, the cut-cell method to the polygone buildings. 
 *
 * @sa Building
 * @sa ParseInterface
 */
class CutBuilding : public Building
{
private:
protected:
  
  ///@{
  /** :document this */
  std::vector<float> xi, yi;
  ///@}

  ///@{
  /** :document this: */
  std::vector<float> xf1, yf1, xf2, yf2;
  ///@}

public:
  
  CutBuilding()
    : Building()
  {
  }

  virtual ~CutBuilding()
  {
  }

  void setCutCellFlags(WINDSGeneralData *WGD, int building_number, float x_min, float x_max, float y_min,
		       float y_max, int i_start, int i_end, int j_start, int j_end, int k_start, int k_end,
		       int k_cut_end, float base_height, float height_eff, std::vector<polyVert> polygonVertices);

  /**
   * :document this:
   *
   * @param face_points :document this:
   * @param index :document this:
   */
  void reorderPoints(std::vector<cutVert> &face_points, int index);

  /**
   * :document this:
   *
   * @param angle :document this:
   * @param face_points :document this:
   */
  void mergeSort(std::vector<float> &angle, std::vector<cutVert> &face_points);


  /**
   * :document this:
   *
   * @param WGD :document this:
   * @param face_points :document this:
   * @param cutcell_index :document this:
   * @param index :document this:
   */
  float calculateArea(WINDSGeneralData *WGD, std::vector<cutVert> &face_points, int cutcell_index, int index);

};
