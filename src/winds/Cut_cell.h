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

/** @file Cut_cell.h */

#pragma once

#include <math.h>
#include <algorithm>
#include "util/Vector3.h"
#include "Edge.h"
#include "Cell.h"
#include "DTEHeightField.h"

class WINDSInputData;
class WINDSGeneralData;

/**
 * @class Cut_cell
 * @brief Designed to store and handle information related to the Cut_cells.
 *
 * :explain what distinguishes a cut_cell from a cell here:
 *
 * @sa Cell
 */
class Cut_cell
{
private:
  const float pi = 4.0f * atan(1.0); /**< pi constant */

public:
  friend class test_CutCell;

  /**
   * This function calculates area fraction coefficients used in the cut-cell method.
   *
   * It takes in a pointer to cell and terrain information (intersection points) and after sorting them for each
   * face of the cell and calculating coefficients, it sets them to the related solver coefficient (e,f,g,h,m,n)
   *
   * @param cells :document this:
   * @param DTEHF :document this:
   * @param WID :document this:
   * @param WGD :document this:
   */
  void calculateCoefficient(Cell *cells, const DTEHeightField *DTEHF, const WINDSInputData *WID, WINDSGeneralData *WGD);

  /**
   * This function takes in intersection points for each face and reorder them based on angle.
   *
   * It fisrt calculates the centroid of points (simple average). Then it reorders points based
   * on their angle start from -180 to 180 degree.
   *
   * @param cut_points :document this:
   * @param index :document this:
   * @param pi :document this:
   */
  void reorderPoints(std::vector<Vector3<float>> &cut_points, int index, float pi);

  /**
   * This function takes in points and their calculated angles and sort them from lowest to
   * largest.
   *
   * @param angle :document this:
   * @param cutPoints :document this:
   */
  void mergeSort(std::vector<float> &angle, std::vector<Vector3<float>> &cutPoints);

private:
  /**
   * This function takes in sorted intersection points and calculates area fraction coefficients
   * based on polygon area formulation. Then it sets them to related solver coefficients.
   *
   * @param cut_points :document this:
   * @param cutcell_index :document this:
   * @param dx :document this:
   * @param dy :document this:
   * @param dz :document this:
   * @param n  :document this:
   * @param m  :document this:
   * @param f  :document this:
   * @param e  :document this:
   * @param h  :document this:
   * @param g  :document this:
   * @param index :document this:
   */
  float calculateArea(std::vector<Vector3<float>> &cut_points, int cutcell_index, float dx, float dy, float dz, std::vector<float> &n, std::vector<float> &m, std::vector<float> &f, std::vector<float> &e, std::vector<float> &h, std::vector<float> &g, int index);

  /**
   * This function uses the edges that form triangles that lie on either the top or bottom of the cell to
   * calculate the terrain area that covers each face.
   *
   * @param terrainPoints the points in the cell that mark a separation of terrain and air
   * @param terrainEdges a list of edges that exist between terrainPoints
   * @param cellIndex the index of the cell (this is needed for the coef)
   * @param dx dimension of the cell in the x direction
   * @param dy dimension of the cell in the y direction
   * @param dz dimension of the cell in the z direction
   * @param location the location of the corner of the cell closest to the origin
   * @param coef the coefficient that should be updated
   * @param isBot states if the area for the bottom or top of the cell should be calculated
   */
  float calculateAreaTopBot(std::vector<Vector3<float>> &terrainPoints,
    const std::vector<Edge<int>> &terrainEdges,
    const int cellIndex,
    const float dx,
    const float dy,
    const float dz,
    Vector3<float> location,
    std::vector<float> &coef,
    const bool isBot);
};
