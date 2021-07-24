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

/**
 * @file Cut_cell.cpp
 * @brief Designed to store and handle information related to the Cut_cells.
 *
 * @sa Cell
 */

#include "Cut_cell.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


void Cut_cell::calculateCoefficient(Cell *cells, const DTEHeightField *DTEHF, const WINDSInputData *WID, WINDSGeneralData *WGD)
{

  std::vector<int> cutcell_index;// Index of cut-cells
  std::vector<Vector3> cut_points;// Intersection points for each face
  std::vector<Edge<int>> terrainEdges;
  Vector3 location;// Coordinates of the left corner of cell face


  cells = new Cell[(WGD->nx - 1) * (WGD->ny - 1) * (WGD->nz - 1)];
  // Get cut-cell indices from terrain function
  cutcell_index = DTEHF->setCells(cells, WGD->nx, WGD->ny, WGD->nz, WGD->dx, WGD->dy, WGD->dz_array, WGD->z_face, WID->simParams->halo_x, WID->simParams->halo_y);

  std::cout << "number of cut cells:" << cutcell_index.size() << "\n";

  // Set icellflag value for terrain cells
#pragma acc parallel loop independent
  for (int i = 0; i < WGD->nx - 1; i++) {
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int k = 1; k < WGD->nz - 1; k++) {
        int icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        if (cells[icell_cent].getIsTerrain()) {
          WGD->icellflag[icell_cent] = 2;
        }
      }
    }
  }

#pragma acc parallel loop independent
  for (auto i = 0; i < cutcell_index.size(); i++) {
    WGD->icellflag[cutcell_index[i]] = 8;
  }

  float S_front, S_behind, S_right, S_left, S_below, S_above;
  float S_cut;
  float ni, nj, nk;
  float solid_V_frac;

  // For all cut cells
  for (int j = 0; j < cutcell_index.size(); j++) {
    S_front = S_behind = S_right = S_left = S_below = S_above = 0.0;
    S_cut = 0.0;
    ni = nj = nk = 0.0;
    solid_V_frac = 0.0;
    location = cells[cutcell_index[j]].getLocationPoints();
    terrainEdges = cells[cutcell_index[j]].getTerrainEdges();
    std::vector<Vector3> terrainPoints = cells[cutcell_index[j]].getTerrainPoints();
    int k = cutcell_index[j] / ((WGD->nx - 1) * (WGD->ny - 1));
    int jjj = (cutcell_index[j] - k * (WGD->nx - 1) * (WGD->ny - 1)) / (WGD->nx - 1);
    int iii = cutcell_index[j] - k * (WGD->nx - 1) * (WGD->ny - 1) - jjj * (WGD->nx - 1);
    // for every face
    for (int i = 0; i < 6; i++) {
      cut_points.clear();
      cut_points = cells[cutcell_index[j]].getFaceFluidPoints(i);

      // place points in local cell space
      if (cut_points.size() > 2) {
        for (int jj = 0; jj < cut_points.size(); jj++) {
          for (int l = 0; l < 3; l++) {
            cut_points[jj][l] = cut_points[jj][l] - location[l];
          }
        }

        // for faces that exist on the side of the cell (not XY planes)
        if (i < 4) {
          reorderPoints(cut_points, i, pi);
          if (i == 0) {
            S_right = calculateArea(cut_points, cutcell_index[j], WGD->dx, WGD->dy, WGD->dz_array[k], WGD->n, WGD->m, WGD->f, WGD->e, WGD->h, WGD->g, i);
          }
          if (i == 1) {
            S_left = calculateArea(cut_points, cutcell_index[j], WGD->dx, WGD->dy, WGD->dz_array[k], WGD->n, WGD->m, WGD->f, WGD->e, WGD->h, WGD->g, i);
          }
          if (i == 2) {
            S_front = calculateArea(cut_points, cutcell_index[j], WGD->dx, WGD->dy, WGD->dz_array[k], WGD->n, WGD->m, WGD->f, WGD->e, WGD->h, WGD->g, i);
          }
          if (i == 3) {
            S_behind = calculateArea(cut_points, cutcell_index[j], WGD->dx, WGD->dy, WGD->dz_array[k], WGD->n, WGD->m, WGD->f, WGD->e, WGD->h, WGD->g, i);
          }
        }
      }
      // for the top and bottom faces of the cell (XY planes)

      if (i == 4) {
        S_below = calculateAreaTopBot(terrainPoints, terrainEdges, cutcell_index[j], WGD->dx, WGD->dy, WGD->dz_array[k], location, WGD->n, true);
      }

      if (i == 5) {
        S_above = calculateAreaTopBot(terrainPoints, terrainEdges, cutcell_index[j], WGD->dx, WGD->dy, WGD->dz_array[k], location, WGD->m, false);
      }

      S_cut = sqrt(pow(S_behind - S_front, 2.0) + pow(S_right - S_left, 2.0) + pow(S_below - S_above, 2.0));

      if (S_cut != 0.0) {
        ni = (S_behind - S_front) / S_cut;
        nj = (S_right - S_left) / S_cut;
        nk = (S_below - S_above) / S_cut;
      }

      if (i == 0) {
        solid_V_frac += (0.0 * (-1) * S_right) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
      }

      if (i == 1) {
        solid_V_frac += (WGD->dy * (1) * S_left) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
      }

      if (i == 2) {
        solid_V_frac += (WGD->dx * (1) * S_front) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
      }

      if (i == 3) {
        solid_V_frac += (0.0 * (-1) * S_behind) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
      }

      if (i == 4) {
        solid_V_frac += (0.0 * (-1) * S_below) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
      }

      if (i == 5) {
        solid_V_frac += (WGD->dz_array[k] * (1) * S_above) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
      }
    }

    if (terrainPoints.size() != 0) {
      solid_V_frac += (((terrainPoints[0][0] - location[0]) * ni * S_cut) + ((terrainPoints[0][1] - location[1]) * nj * S_cut) + ((terrainPoints[0][2] - location[2]) * nk * S_cut)) / (3 * WGD->dx * WGD->dy * WGD->dz_array[k]);
    }

    WGD->terrain_volume_frac[cutcell_index[j]] -= solid_V_frac;

    WGD->ni[cutcell_index[j]] = ni;
    WGD->nj[cutcell_index[j]] = nj;
    WGD->nk[cutcell_index[j]] = nk;

    if (WGD->terrain_volume_frac[cutcell_index[j]] < 0.0) {
      WGD->terrain_volume_frac[cutcell_index[j]] = 0.0;
    }
  }
}


void Cut_cell::reorderPoints(std::vector<Vector3> &cut_points, int index, float pi)
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


void Cut_cell::mergeSort(std::vector<float> &angle, std::vector<Vector3> &cutPoints)
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

/*
{

    std::vector<float> angle_temp (cut_points.size(), 0.0);
    std::vector<float> angle_max (cut_points.size(), 0.0);
    std::vector<Vector3<float>> cut_points_temp;
    std::vector<int> imax (cut_points.size(),0);

    cut_points_temp = cut_points;

    for (int i=0; i<cut_points.size(); i++)
    {
        imax[i] = i;
        angle_max[i] = -180;
        angle_temp[i] = angle[i];
    }

    for (int i=0; i<cut_points.size(); i++)
    {
        for (int j=0; j<cut_points.size(); j++)
        {
            if (angle[j] > angle_max[i]){
                angle_max[i] = angle[j];
                imax[i] = j;
            }
        }
        angle[imax[i]] = -999;
    }

    for (int i=0; i<cut_points.size(); i++)
    {
        cut_points[i][0] = cut_points_temp[imax[cut_points.size()-1-i]][0];
        cut_points[i][1] = cut_points_temp[imax[cut_points.size()-1-i]][1];
        cut_points[i][2] = cut_points_temp[imax[cut_points.size()-1-i]][2];
        angle[i] = angle_temp[imax[cut_points.size()-1-i]];
    }

}*/

float Cut_cell::calculateArea(std::vector<Vector3> &cut_points, int cutcell_index, float dx, float dy, float dz, std::vector<float> &n, std::vector<float> &m, std::vector<float> &f, std::vector<float> &e, std::vector<float> &h, std::vector<float> &g, int index)
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
  if (coeff >= 0) {
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
    /*if (index == 4)
    {
        S = (1.0 - coeff)*(dx*dy);
        n[cutcell_index] = coeff;
    }
    if (index == 5)
    {
        S = (1.0 - coeff)*(dx*dy);
        m[cutcell_index] = coeff;
    }*/
  }
  return S;
}


float Cut_cell::calculateAreaTopBot(std::vector<Vector3> &terrainPoints,
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
  for (int i = 0; i < terrainPoints.size(); i++)
    if (terrainPoints[i][2] > faceHeight - 0.00001f && terrainPoints[i][2] < faceHeight + 0.00001f)
      pointsOnFace.push_back(i);

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

  // when on the bottom, the area of triangles is the area of the air, so subtract it from the face size
  if (!isBot)
    area = dx * dy - area;

  coef[cellIndex] = area / (dx * dy);
  S = (1.0 - coef[cellIndex]) * (dx * dy);
  return S;
}
