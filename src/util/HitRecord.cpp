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
 ***************************************************************************/

/**
 * @file HitRecord.cpp
 * @brief Used to store information about intersections.
 *
 * @note Can add other information about the BVH node it hits as needed.
 *
 * @sa Vector3Float
 * @sa BVH
 */

#include "HitRecord.h"

HitRecord::HitRecord()
  : isHit(false), hitNode(nullptr), hitDist(0.0), t(0.0)
{
}

HitRecord::HitRecord(void *hitNode, bool isHit)
  : t(0.0)
{
  this->hitNode = hitNode;
  this->isHit = isHit;
  hitDist = -1 * (std::numeric_limits<float>::infinity());
}

HitRecord::HitRecord(void *hitNode, bool isHit, float hitDist)
  : t(0.0)
{
  this->hitNode = hitNode;
  this->isHit = isHit;
  this->hitDist = hitDist;
}

void *HitRecord::getHitNode() { return hitNode; }
float HitRecord::getHitDist() { return hitDist; }
bool HitRecord::getIsHit() { return isHit; }
