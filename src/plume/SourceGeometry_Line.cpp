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

/** @file SourceLine.cpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "SourceGeometry_Line.hpp"

void SourceGeometry_Line::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{
  if (posX_0 < domainXstart || posX_0 > domainXend) {
    std::cerr << "[ERROR] \t SourceGeometry_Line::checkPosInfo: \n\t\t input posX_0 is outside of domain! posX_0 = \"" << posX_0
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (posY_0 < domainYstart || posY_0 > domainYend) {
    std::cerr << "[ERROR] \t SourceGeometry_Line::checkPosInfo: \n\t\t input posY_0 is outside of domain! posY_0 = \"" << posY_0
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (posZ_0 < domainZstart || posZ_0 > domainZend) {
    std::cerr << "[ERROR] \t SourceGeometry_Line::checkPosInfo: \n\t\t input posZ_0 is outside of domain! posZ_0 = \"" << posZ_0
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }

  if (posX_1 < domainXstart || posX_1 > domainXend) {
    std::cerr << "[ERROR] \t SourceGeometry_Line::checkPosInfo: \n\t\t input posX_1 is outside of domain! posX_1 = \"" << posX_1
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (posY_1 < domainYstart || posY_1 > domainYend) {
    std::cerr << "[ERROR] \t SourceGeometry_Line::checkPosInfo: \n\t\t input posY_1 is outside of domain! posY_1 = \"" << posY_1
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (posZ_1 < domainZstart || posZ_1 > domainZend) {
    std::cerr << "[ERROR] \t SourceGeometry_Line::checkPosInfo: \n\t\t input posZ_1 is outside of domain! posZ_1 = \"" << posZ_1
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}

void SourceGeometry_Line::setInitialPosition(double &x, double &y, double &z)
{
  // generate random point on line between m_pt0 and m_pt1
  double diffX = posX_1 - posX_0;
  double diffY = posY_1 - posY_0;
  double diffZ = posZ_1 - posZ_0;
  float t = drand48();
  x = posX_0 + t * diffX;
  y = posY_0 + t * diffY;
  z = posZ_0 + t * diffZ;
}