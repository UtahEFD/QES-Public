/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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

/** @file Mesh.h */

#pragma once

#include "Triangle.h"
#include "BVH.h"
#include "SphereDirections.h"
#include "Ray.h"
#include "HitRecord.h"
// #include "OptixRayTrace.h"

#include <limits>
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <iostream>

using std::vector;

/**
 * @class Mesh
 * @brief Serves as a container for a BVH of Triangle objects that
 * represents a connected collection of Triangles.
 *
 * @sa BVH
 * @sa Triangle
 * @sa SphereDirections
 * @sa Ray
 * @sa HitRecord
 */
class Mesh
{
public:
  BVH *triangleBVH; /**< BVH of Triangle objects */

  int mlSampleRate; /**< :document this: */

  // temp var for Optix
  // OptixRayTrace *optixRayTracer;
  vector<Triangle *> optixTris;

  /**
   * Creates a BVH out of a list of Triangles.
   *
   * @param tris List of triangles.
   */
  Mesh(const vector<Triangle *> &tris)
    : mlSampleRate(100)
  {
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "[Mesh]\t\t Initialization of triangular mesh...\n";

    auto start = std::chrono::high_resolution_clock::now();

    this->triangleBVH = BVH::createBVH(tris);

    // temp var for Optix
    //  this->optixRayTracer = new OptixRayTrace(tris);
    optixTris = tris;
    // temp var for getting the list of triangles
    trisList = tris;

    auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
    std::chrono::duration<float> elapsed_cut = finish - start;
    std::cout << "\t\t elapsed time: " << elapsed_cut.count() << " s\n";
  }

  /**
   * Gets the height from a location on the xy-plane
   * to a triangle in the BVH.
   *
   * @param x x-position
   * @param y y-position
   * @return distance to the triangle directly above the point.
   */
  float getHeight(float x, float y);


  /**
   * Calculates the mixing length for all fluid objects.
   *
   *@param dimX Domain info in the x plane
   *@param dimY Domain info in the y plane
   *@param dimZ Domain info in the z plane
   *@param dx Grid info in the x plane
   *@param dy Grid info in the y plane
   *@param dz Grid info in the z plane
   *@param icellflag Cell type
   *@param mixingLengths Array of mixinglengths for all cells that will be updated
   */
  void calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, vector<double> &mixingLengths);

  /**
   * :document this:
   *@param dimX Domain info in the x plane
   *@param dimY Domain info in the y plane
   *@param dimZ Domain info in the z plane
   *@param dx Grid info in the x plane
   *@param dy Grid info in the y plane
   *@param dz Grid info in the z plane
   *@param icellflag Cell type
   *@param mixingLengths Array of mixinglengths for all cells that will be updated
   */
  void tempOPTIXMethod(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, vector<double> &mixingLengths);

  std::vector<Triangle *> getTris() const
  {
    return trisList;
  }

private:
  std::vector<Triangle *> trisList; /**< Temporary variable for getting th elist of Triangle objects through the mesh */
};
