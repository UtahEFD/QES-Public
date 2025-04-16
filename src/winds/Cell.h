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

/** @file Cell.h */

#pragma once

#include "util/Vector3Float.h"
#include "Edge.h"
#include <vector>

enum cellType : int { air_CT,
                      terrain_CT };
enum cellFace : int { faceXZNeg_CF,
                      faceXZPos_CF,
                      faceYZNeg_CF,
                      faceYZPos_CF,
                      faceXYNeg_CF,
                      faceXYPos_CF };

/**
 * @class Cell
 * @brief Holds information about cell data.
 *
 * Mostly pertains to what entities exit in the cell, and what coordinate
 * points form the terrain in the cell and what edges connect
 * the points (if it exists). If a cell is a Cut_cell it will also
 * have a set of points that lie on each face.
 *
 * @sa Cut_cell
 */
class Cell
{
private:
  ///@{
  /** boolean property */
  bool isAir, isTerrain, isCutCell;

  ///@}

  // std::vector<Vector3Float> terrainPoints; /**< List of terrain points */
  // std::vector<Edge<int>> terrainEdges; /**< List of edges that connect the terrain points */
  // std::vector<Vector3Float> fluidFacePoints[6]; /**< :document this: */
  // Vector3Float location; /**< XYZ location of the cell */
  // Vector3Float dimensions; /**< Size of the cell in xyz directions */
public:
  /**
   * @return true if air exists, else false
   * @brief Returns true if air exists in the cell.
   */
  bool getIsAir() { return isAir; }

  /**
   * @return true if terrain exists, else false
   * @brief Returns true if terrain exists in the cell.
   */
  bool getIsTerrain() { return isTerrain; }

  /**
   * @return true if both terrain and air is in the cell, else false
   * @brief Returns true if terrain is partially in the cell.
   */
  bool getIsCutCell() { return isCutCell; }

  /**
   * @return the list of terrain points
   * @brief Returns a list of coordinate points that form the terrain in the cell.
   */
  // std::vector<Vector3Float> getTerrainPoints() { return terrainPoints; }

  /**
   * @return the list of edges connecting terrain points
   * @brief Returns a list of edges that connect the terrain points in the cell.
   */

  // std::vector<Edge<int>> getTerrainEdges() { return terrainEdges; }

  /**
   * Defaults all entity existances values to false, and has no terrain points.
   */
  Cell()
    : isAir(false), isTerrain(false), isCutCell(false)
  {}

  /**
   * Takes in a specific entity that totally fills the cell, set's that value
   * to true.
   *
   * @param type_CT what is filling the cell
   * @param locationN the position of the corner closest to the origin
   * @param dimensionsN the size of the cell in the xyz directions
   */
  // Cell(const int type_CT, Vector3Float locationN, Vector3Float dimensionsN);

  /**
   * Takes in a list of terrain points that exist in the cell separating where
   * the terrain and air exist in the cell, sets isCutCell to true.
   *
   * @param points a list of points that form the cut.
   * @param edges a list of edges that form the terrain
   * @param intermed a collection of intermediate points between corners that rest on the top and bottom of the cell
   * @param locationN the position of the corner closest to the origin
   * @param dimensionsN the size of the cell in the xyz directions
   */
  // Cell(std::vector<Vector3Float> &points, int intermed[4][4][2], Vector3Float locationN, Vector3Float dimensionsN);
  // Cell(std::vector<Vector3Float> &points, std::vector<Edge<int>> &edges, int intermed[4][4][2], Vector3Float locationN, Vector3Float dimensionsN);


  /**
   * @return the fluidFacePoint at the index provided
   * @param index the index of the face to be returned (cellFace enum)
   * @brief  Returns a vector of points that lie on the specified face.
   */
  // std::vector<Vector3Float> getFaceFluidPoints(const int index) { return fluidFacePoints[index % 6]; }


  /**
   * @return the location of the cell
   * @brief Returns the xyz location of the cell from the corner closest to the origin.
   */
  // Vector3Float getLocationPoints() { return location; }
};
