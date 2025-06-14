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

/** @file HitRecord.h */

#pragma once

#ifndef HR_H
#define HR_H

#include "Vector3Float.h"
#include <limits>

/**
 * @class HitRecord
 * @brief Used to store information about intersections.
 *
 * @note Can add other information about the BVH node it hits as needed.
 *
 * @sa Vector3Float
 * @sa BVH
 */
class HitRecord
{
public:
  bool isHit; /**< :document this: */
  void *hitNode; /**< Reference to BVH node that was hit */
  float hitDist; /**< Distance from ray origin to hit point */
  float t; /**< :document this: */
  Vector3Float endpt; /**< The intersection point */
  Vector3Float n; /**< The normal to surface at intersection point */

  HitRecord();
  HitRecord(void *hitNode, bool isHit);
  HitRecord(void *hitNode, bool isHit, float hitDist);

  void *getHitNode();
  float getHitDist();
  bool getIsHit();
};

#endif
