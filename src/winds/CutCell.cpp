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
 * @file CutCell.cpp
 * @brief Designed to store and handle information related to the Cut_cells.
 *
 * @sa Cell
 */

#include "CutCell.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


void CutCell::reorderPoints(std::vector<Vector3> &cut_points, int index, float pi)
{

  Vector3 centroid;
  std::vector<float> angle(cut_points.size(), 0.0);
  Vector3 sum;

  sum[0] = 0;
  sum[1] = 0;
  sum[2] = 0;

  // Calculate centroid of points
  for (int i = 0; i < cut_points.size(); i++) {
    sum[0] += cut_points[i][0];
    sum[1] += cut_points[i][1];
    sum[2] += cut_points[i][2];
  }

  centroid[0] = sum[0] / cut_points.size();
  centroid[1] = sum[1] / cut_points.size();
  centroid[2] = sum[2] / cut_points.size();

  // Calculate angle between each point and centroid
  for (int i = 0; i < cut_points.size(); i++) {
    if (index == 2 || index == 3) {
      angle[i] = (180 / pi) * atan2((cut_points[i][2] - centroid[2]), (cut_points[i][1] - centroid[1]));
    }
    if (index == 0 || index == 1) {
      angle[i] = (180 / pi) * atan2((cut_points[i][2] - centroid[2]), (cut_points[i][0] - centroid[0]));
    }
    if (index == 4 || index == 5) {
      angle[i] = (180 / pi) * atan2((cut_points[i][1] - centroid[1]), (cut_points[i][0] - centroid[0]));
    }
  }
  // Call sort to sort points based on the angles (from -180 to 180)
  mergeSort(angle, cut_points);
}


void CutCell::mergeSort(std::vector<float> &angle, std::vector<Vector3> &cutPoints)
{
  // if the size of the array is 1, it is already sorted
  if (angle.size() == 1)
    return;

  // make left and right sides of the data
  std::vector<float> angleL, angleR;
  std::vector<Vector3> cutPointsL, cutPointsR;

  angleL.resize(angle.size() / 2);
  angleR.resize(angle.size() - angle.size() / 2);
  cutPointsL.resize(cutPoints.size() / 2);
  cutPointsR.resize(cutPoints.size() - cutPoints.size() / 2);

  // copy data from the main data set to the left and right children
  int lC = 0, rC = 0;
  for (unsigned int i = 0; i < angle.size(); i++) {
    if (i < angle.size() / 2) {
      angleL[lC] = angle[i];
      cutPointsL[lC++] = cutPoints[i];
    } else {
      angleR[rC] = angle[i];
      cutPointsR[rC++] = cutPoints[i];
    }
  }

  // recursively sort the children
  mergeSort(angleL, cutPointsL);
  mergeSort(angleR, cutPointsR);

  // compare the sorted children to place the data into the main array
  lC = rC = 0;
  for (unsigned int i = 0; i < cutPoints.size(); i++) {
    if (rC == angleR.size() || (lC != angleL.size() && angleL[lC] < angleR[rC])) {
      angle[i] = angleL[lC];
      cutPoints[i] = cutPointsL[lC++];
    } else {
      angle[i] = angleR[rC];
      cutPoints[i] = cutPointsR[rC++];
    }
  }

  return;
}

float CutCell::calculateArea(std::vector<Vector3> &cut_points, int cutcell_index, float dx, float dy, float dz, std::vector<float> &n, std::vector<float> &m, std::vector<float> &f, std::vector<float> &e, std::vector<float> &h, std::vector<float> &g, int index)
{
  float S = 0.0;
  float coeff = 0;
  if (cut_points.size() != 0) {
    /// calculate area fraction coeeficient for each face of the cut-cell
    for (int i = 0; i < cut_points.size() - 1; i++) {
      coeff += (0.5 * (cut_points[i + 1][1] + cut_points[i][1]) * (cut_points[i + 1][2] - cut_points[i][2])) / (dy * dz) + (0.5 * (cut_points[i + 1][0] + cut_points[i][0]) * (cut_points[i + 1][2] - cut_points[i][2])) / (dx * dz) + (0.5 * (cut_points[i + 1][0] + cut_points[i][0]) * (cut_points[i + 1][1] - cut_points[i][1])) / (dx * dy);
    }

    coeff += (0.5 * (cut_points[0][1] + cut_points[cut_points.size() - 1][1]) * (cut_points[0][2] - cut_points[cut_points.size() - 1][2])) / (dy * dz) + (0.5 * (cut_points[0][0] + cut_points[cut_points.size() - 1][0]) * (cut_points[0][2] - cut_points[cut_points.size() - 1][2])) / (dx * dz) + (0.5 * (cut_points[0][0] + cut_points[cut_points.size() - 1][0]) * (cut_points[0][1] - cut_points[cut_points.size() - 1][1])) / (dx * dy);
  }

  if (coeff <= 0.05) {
    coeff = 0.0;
  } else if (coeff > 1.0) {
    coeff = 1.0;
  }

  if (index == 3) {
    S = (1.0 - coeff) * (dy * dz);
    f[cutcell_index] = coeff;
  }
  if (index == 2) {
    S = (1.0 - coeff) * (dy * dz);
    e[cutcell_index] = coeff;
  }
  if (index == 0) {
    S = (1.0 - coeff) * (dx * dz);
    h[cutcell_index] = coeff;
  }
  if (index == 1) {
    S = (1.0 - coeff) * (dx * dz);
    g[cutcell_index] = coeff;
  }
  if (index == 4) {
    S = coeff * (dx * dy);
    n[cutcell_index] = 1.0 - coeff;
  }
  if (index == 5) {
    S = coeff * (dx * dy);
    m[cutcell_index] = 1.0 - coeff;
  }
  return S;
}


float CutCell::calculateAreaTopBot(std::vector<Vector3> &terrainPoints,
                                    const std::vector<Edge<int>> &terrainEdges,
                                    const int cellIndex,
                                    const float dx,
                                    const float dy,
                                    const float dz,
                                    Vector3 location,
                                    std::vector<float> &coef,
                                    const bool isBot)
{
  float S = 0.0;
  float area = 0.0f;
  std::vector<int> pointsOnFace;
  std::vector<Vector3> listOfTriangles;// each point is a vector3, the triangle is 3 points
  float faceHeight = location[2] + (isBot ? 0.0f : dz);// face height is 0 if we are on the bottom, otherwise add dz_array

  // find all points in the terrain on this face
  for (int i = 0; i < terrainPoints.size(); i++) {
    if (terrainPoints[i][2] >= faceHeight) {
      pointsOnFace.push_back(i);
    }
  }

  // find list of triangles
  if (pointsOnFace.size() > 2) {
    for (int a = 0; a < pointsOnFace.size() - 2; a++)
      for (int b = a + 1; b < pointsOnFace.size() - 1; b++)
        for (int c = b + 1; c < pointsOnFace.size(); c++) {
          // triangle is on face if a,b a,c b,c edges all exist (note edges are reversable)
          //  a|b|c is the index in pointsOnFace, which is the index in terrainPoint that we are representing.
          Edge<int> abEdge(pointsOnFace[a], pointsOnFace[b]), acEdge(pointsOnFace[a], pointsOnFace[c]),
            bcEdge(pointsOnFace[b], pointsOnFace[c]);
          if ((std::find(terrainEdges.begin(), terrainEdges.end(), abEdge) != terrainEdges.end()) && (std::find(terrainEdges.begin(), terrainEdges.end(), acEdge) != terrainEdges.end()) && (std::find(terrainEdges.begin(), terrainEdges.end(), bcEdge) != terrainEdges.end())) {
            listOfTriangles.push_back(Vector3(terrainPoints[pointsOnFace[a]][0],
                                              terrainPoints[pointsOnFace[a]][1],
                                              terrainPoints[pointsOnFace[a]][2]));
            listOfTriangles.push_back(Vector3(terrainPoints[pointsOnFace[b]][0],
                                              terrainPoints[pointsOnFace[b]][1],
                                              terrainPoints[pointsOnFace[b]][2]));
            listOfTriangles.push_back(Vector3(terrainPoints[pointsOnFace[c]][0],
                                              terrainPoints[pointsOnFace[c]][1],
                                              terrainPoints[pointsOnFace[c]][2]));
          }
        }
  }

  // for all triangles, add the area to the total
  for (int t = 0; t < listOfTriangles.size(); t = t + 3) {
    Vector3 a = listOfTriangles[t];
    Vector3 b = listOfTriangles[t + 1];
    Vector3 c = listOfTriangles[t + 2];

    // move to local space
    for (int d = 0; d < 3; d++) {
      a[d] -= location[d];
      b[d] -= location[d];
      c[d] -= location[d];
    }

    float tempArea = (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0f;
    area += (tempArea < 0.0f ? tempArea * -1.0f : tempArea);
  }

  area = 1.0 - (area / (dx * dy));

  if (area <= 0.05) {
    area = 0.0;
  }

  coef[cellIndex] = area;
  S = (1.0 - coef[cellIndex]) * (dx * dy);

  return S;
}
