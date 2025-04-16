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
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file SourcePoint.cpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "PI_SourceGeometry_Point.hpp"
#define _USE_MATH_DEFINES
#include <cmath>

SourceComponent *PI_SourceGeometry_Point::create(QESDataTransport &data)
{
  // return new SourceGeometryPoint(this);
  return new SourceGeometryPoint({ m_posX, m_posY, m_posZ });
}

/*
void PI_SourceGeometry_Point::checkPosInfo(const float &domainXstart,
                                           const float &domainXend,
                                           const float &domainYstart,
                                           const float &domainYend,
                                           const float &domainZstart,
                                           const float &domainZend)
{
  if (posX < domainXstart || posX > domainXend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_Point::checkPosInfo: \n\t\t input posX is outside of domain! posX = \"" << posX
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (posY < domainYstart || posY > domainYend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_Point::checkPosInfo: \n\t\t input posY is outside of domain! posY = \"" << posY
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (posZ < domainZstart || posZ > domainZend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_Point::checkPosInfo: \n\t\t input posZ is outside of domain! posZ = \"" << posZ
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}

void PI_SourceGeometry_Point::setInitialPosition(vec3 &p)
{
  // set initial position
  p._1 = posX;
  p._2 = posY;
  p._3 = posZ;
}
*/