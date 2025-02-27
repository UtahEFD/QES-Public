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

/** @file Triangle.h */

#pragma once

#include "Vector3Float.h"
#include "Ray.h"
#include "HitRecord.h"
#include <cmath>


#define LOWEST_OF_THREE(x, y, z) ((x) <= (y) && (x) <= (z) ? (x) : ((y) <= (x) && (y) <= (z) ? (y) : (z)))
#define HIGHEST_OF_THREE(x, y, z) ((x) >= (y) && (x) >= (z) ? (x) : ((y) >= (x) && (y) >= (z) ? (y) : (z)))

/**
 * @class Triangle
 * @brief Represents a triangle made of 3 points, each with an x,y,z location.
 */
class Triangle
{
public:
  Vector3Float a, b, c;
  Vector3Float n;

  Triangle()
    : a(0.0, 0.0, 0.0),
      b(0.0, 0.0, 0.0),
      c(0.0, 0.0, 0.0),
      n(0.0, 0.0, 0.0)
  {
  }

  Triangle(const Vector3Float &aN, const Vector3Float &bN, const Vector3Float &cN)
    : a(aN), b(bN), c(cN)
  {
    Vector3Float v(bN - aN);
    Vector3Float w(cN - aN);
    n = v.cross(w);
    n = n / n.length();
  }

  /**
   * Uses a vertical ray cast from point x y at height 0 with barycentric interpolation to
   * determine if the ray hits inside this triangle.
   *
   * @param x x-location
   * @param y y-location
   * @return the length of the ray before intersection, if no intersection, -1 is returned
   */
  float getHeightTo(float x, float y);


  /**
   * Gets the minimum and maximum values in the x y and z dimensions.
   *
   * @param xmin lowest value in the x dimension
   * @param xmax highest value in the x dimension
   * @param ymin lowest value in the y dimension
   * @param ymax highest value in the y dimension
   * @param zmin lowest value in the z dimension
   * @param zmax highest value in the z dimension
   */
  void getBoundaries(float &xmin, float &xmax, float &ymin, float &ymax, float &zmin, float &zmax);


  /**
   * Determines if a ray hit the triangle and updates the hit record.
   *
   * @param ray the ray to be checked for intersection
   * @param rec the HitRecord to be updated
   */
  bool rayTriangleIntersect(const Ray &ray, HitRecord &rec, float t0, float t1);
};
