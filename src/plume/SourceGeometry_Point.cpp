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

#include "SourceGeometry_Point.hpp"
#include "winds/WINDSGeneralData.h"
#define _USE_MATH_DEFINES
#include <cmath>
// #include "Interp.h"

void SourceGeometry_Point::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{
  if (posX < domainXstart || posX > domainXend) {
    std::cerr << "[ERROR] \t SourceGeometry_Point::checkPosInfo: \n\t\t input posX is outside of domain! posX = \"" << posX
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (posY < domainYstart || posY > domainYend) {
    std::cerr << "[ERROR] \t SourceGeometry_Point::checkPosInfo: \n\t\t input posY is outside of domain! posY = \"" << posY
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (posZ < domainZstart || posZ > domainZend) {
    std::cerr << "[ERROR] \t SourceGeometry_Point::checkPosInfo: \n\t\t input posZ is outside of domain! posZ = \"" << posZ
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}

// template <class typeid(parType).name()>
void SourceGeometry_Point::setInitialPosition(Particle *ptr)
{
  // set initial position
  ptr->xPos_init = posX;
  ptr->yPos_init = posY;
  ptr->zPos_init = posZ;
}
