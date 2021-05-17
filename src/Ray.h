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

/** @file Ray.h */

#pragma once

#include "Vec3D.h"

/**
 * @class Ray
 * @brief Basic definition of a ray.
 */
class Ray
{
private:
  float origin_x, origin_y, origin_z;
  Vec3D dirVec;

public:
  Ray(float o_x, float o_y, float o_z, Vec3D &dVec)
    : origin_x(o_x), origin_y(o_y), origin_z(o_z), dirVec(dVec)
  {
  }

  Ray(float o_x, float o_y, float o_z)
    : origin_x(o_x), origin_y(o_y), origin_z(o_z)
  {
    dirVec[0] = 0.0;
    ;
    dirVec[1] = 0.0;
    dirVec[2] = 1.0;
  }

  ~Ray() {}

  float getOriginX() const { return origin_x; }
  float getOriginY() const { return origin_y; }
  float getOriginZ() const { return origin_z; };

  Vec3D getDirection() const { return dirVec; }

  void setDir(const Vec3D &dir)
  {
    dirVec[0] = dir[0];
    dirVec[1] = dir[1];
    dirVec[2] = dir[2];
  }
};
