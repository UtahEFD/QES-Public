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
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

class PolyBuilding;


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
 * In this class, the cut-cell method to the polygon buildings. 
 *
 */
class CutBuilding
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
  {
  }

  virtual ~CutBuilding()
  {
  }

  /**
   * This function calculates the cut-cell area fraction coefficients, the building volume fraction,
   * and the cut face normal unit vectors, and sets up the solver coefficients of each face of the cell.
   *
   * @param WGD Winds general data class pointer
   * @param PolyB polygon building class pointer
   * @param building_number number of the building processing
   */
  void setCutCellFlags(WINDSGeneralData *WGD, const PolyBuilding *PolyB, int building_number);

  /**
   * This function reorders the solid points of a face (passed to it) in a counter-clock wise order. 
   *
   * @param face_points list of solid points on a face
   * @param index index of the face that the points belong to
   */
  void reorderPoints(std::vector<cutVert> &face_points, int index);

  /**
   * This function sorts (based on a merge sort algorithm) the solid points of a face (passed to it) based on their angles.
   *
   * @param angle list of angles of each point on a face
   * @param face_points list of solid points on a face
   */
  void mergeSort(std::vector<float> &angle, std::vector<cutVert> &face_points);


  /**
   * This calculates the area fraction coefficients of a face and sets up the solver coefficient related to the face.
   *
   * @param WGD Winds general data class pointer
   * @param face_points list of solid points on a face
   * @param cutcell_index index of the cut-cell
   * @param index index of the face that the points belong to 
   */
  float calculateArea(WINDSGeneralData *WGD, std::vector<cutVert> &face_points, int cutcell_index, int index);

};
