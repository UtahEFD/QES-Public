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

/** @file SourceLine.cpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "Random.h"
#include "PI_SourceGeometry_Line.hpp"
#include "PLUMEGeneralData.h"

SourceComponent *PI_SourceGeometry_Line::create(QESDataTransport &data)
{
  return new SourceGeometryLine({ m_posX_0, m_posY_0, m_posZ_0 },
                                { m_posX_1, m_posY_1, m_posZ_1 });
}

/*
void PI_SourceGeometry_Line::checkPosInfo(const float &domainXstart,
                                          const float &domainXend,
                                          const float &domainYstart,
                                          const float &domainYend,
                                          const float &domainZstart,
                                          const float &domainZend)
{
  if (posX_0 < domainXstart || posX_0 > domainXend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_Line::checkPosInfo: \n\t\t input posX_0 is outside of domain! posX_0 = \"" << posX_0
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (posY_0 < domainYstart || posY_0 > domainYend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_Line::checkPosInfo: \n\t\t input posY_0 is outside of domain! posY_0 = \"" << posY_0
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (posZ_0 < domainZstart || posZ_0 > domainZend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_Line::checkPosInfo: \n\t\t input posZ_0 is outside of domain! posZ_0 = \"" << posZ_0
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }

  if (posX_1 < domainXstart || posX_1 > domainXend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_Line::checkPosInfo: \n\t\t input posX_1 is outside of domain! posX_1 = \"" << posX_1
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (posY_1 < domainYstart || posY_1 > domainYend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_Line::checkPosInfo: \n\t\t input posY_1 is outside of domain! posY_1 = \"" << posY_1
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (posZ_1 < domainZstart || posZ_1 > domainZend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_Line::checkPosInfo: \n\t\t input posZ_1 is outside of domain! posZ_1 = \"" << posZ_1
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}

void PI_SourceGeometry_Line::setInitialPosition(vec3 &p)
{
  // generate random point on line between m_pt0 and m_pt1
  float diffX = posX_1 - posX_0;
  float diffY = posY_1 - posY_0;
  float diffZ = posZ_1 - posZ_0;

  Random prng;
  float t = prng.uniRan();

  p._1 = posX_0 + t * diffX;
  p._2 = posY_0 + t * diffY;
  p._3 = posZ_0 + t * diffZ;
}
*/