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

/**
 * @file Cell.cpp
 * @brief Holds information about cell data.
 *
 * Mostly pertains to what entities exit in the cell, and what coordinate
 * points form the terrain in the cell and what edges connect
 * the points (if it exists). If a cell is a Cut_cell it will also
 * have a set of points that lie on each face.
 *
 * @sa Cut_cell
 */

#include "Cell.h"

/* OBSOLETE CODE
Cell::Cell()
{
  isAir = isCutCell = isTerrain = false;
  terrainPoints.clear();
  // terrainEdges.clear();
  location = Vector3(0.0f, 0.0f, 0.0f);
  dimensions = Vector3(1.0f, 1.0f, 1.0f);
}

Cell::Cell(const int type_CT, const Vector3 locationN, const Vector3 dimensionsN)
{
  isAir = isCutCell = isTerrain = false;
  terrainPoints.clear();

  if (type_CT == air_CT)
    isAir = true;
  else if (type_CT == terrain_CT)
    isTerrain = true;
  terrainPoints.clear();
  // terrainEdges.clear();
  location = locationN;
  dimensions = dimensionsN;
}

//Cell::Cell(std::vector<Vector3> &points, std::vector<Edge<int>> &edges, int intermed[4][4][2], Vector3 locationN, Vector3 dimensionsN)
Cell::Cell(std::vector<Vector3> &points, int intermed[4][4][2], Vector3 locationN, Vector3 dimensionsN)
{
  isTerrain = isAir = isCutCell = true;
  terrainPoints.clear();
  for (size_t i = 0; i < points.size(); i++)
    terrainPoints.push_back(points[i]);
  //for (int i = 0; i < edges.size(); i++)
  // terrainEdges.push_back(edges[i]);
  location = locationN;
  dimensions = dimensionsN;


  // set fluid points for the XZ and YZ faces
  for (int i = 0; i < 4; i++) {
    int firstC, secondC;
    if (i == 0) {
      firstC = 0;
      secondC = 3;
    } else if (i == 1) {
      firstC = 1;
      secondC = 2;
    } else if (i == 2) {
      firstC = 2;
      secondC = 3;
    } else {
      firstC = 0;
      secondC = 1;
    }

    if (points[firstC][2] > location[2] + dimensions[2]
        && points[secondC][2] > location[2] + dimensions[2]) {
      fluidFacePoints[i].clear();
    } else// if here, then this face is cut
    {
      if (points[firstC][2] < location[2] + dimensions[2]) {
        fluidFacePoints[i].push_back(points[firstC]);
        fluidFacePoints[i].push_back(Vector3(points[firstC][0], points[firstC][1], location[2] + dimensions[2]));
      } else if (intermed[firstC][secondC][1] == -1)
        fluidFacePoints[i].push_back(Vector3(points[firstC][0], points[firstC][1], location[2] + dimensions[2]));

      if (points[secondC][2] < location[2] + dimensions[2]) {
        fluidFacePoints[i].push_back(points[secondC]);
        fluidFacePoints[i].push_back(Vector3(points[secondC][0], points[secondC][1], location[2] + dimensions[2]));
      }

      for (int j = 0; j < 2; j++)
        if (intermed[firstC][secondC][j] != -1)
          fluidFacePoints[i].push_back(points[intermed[firstC][secondC][j]]);

      if (fluidFacePoints[i].size() <= 2)
        fluidFacePoints[i].clear();
    }
  }

  for (int i = 4; i < 6; i++) {
    for (int j = 0; j < 4; j++)
      if ((i == 4) ? points[j][2] <= location[2] :// if this is the bottom, we check if the corner is on the floor
            points[j][2] < location[2] + dimensions[2])// else we check if it's under the ceiling
        fluidFacePoints[i].push_back(points[j]);

    for (int first = 0; first < 3; first++)
      for (int second = first + 1; second < 4; second++)
        if (first != 1 || second != 3)
          if (intermed[first][second][i - 4] != -1)
            fluidFacePoints[i].push_back(points[intermed[first][second][i - 4]]);
    if (fluidFacePoints[i].size() <= 2)
      fluidFacePoints[i].clear();
  }
}

*/
