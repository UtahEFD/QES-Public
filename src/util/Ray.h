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

/** @file Ray.h */

#pragma once

#include "Vector3.h"

/**
 * @class Ray
 * @brief Basic definition of a ray.
 */
class Ray
{
private:
  Vector3 origVec;
  Vector3 dirVec;

public:
  Ray(float o_x, float o_y, float o_z, Vector3 &dVec)
    : origVec(o_x, o_y, o_z), dirVec(dVec)
  {
  }
  Ray(Vector3 &oVec, Vector3 &dVec)
    : origVec(oVec), dirVec(dVec)
  {
  }

  Ray(float o_x, float o_y, float o_z)
    : origVec(o_x, o_y, o_z)
  {
    dirVec[0] = 0.0;
    dirVec[1] = 0.0;
    dirVec[2] = 1.0;
  }

  ~Ray() {}

  float getOriginX() const { return origVec[0]; }
  float getOriginY() const { return origVec[1]; }
  float getOriginZ() const { return origVec[2]; };

  Vector3 getOrigin() const { return origVec; }
  Vector3 getDirection() const { return dirVec; }

  void setOrigin(const Vector3 &orig)
  {
    origVec = orig;
  }
  void setDir(const Vector3 &dir)
  {
    dirVec = dir;
  }
};