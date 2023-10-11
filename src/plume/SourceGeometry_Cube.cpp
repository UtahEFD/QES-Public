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

/** @file SourceCube.cpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "SourceGeometry_Cube.hpp"
#include "winds/WINDSGeneralData.h"
// #include "Interp.h"

void SourceGeometry_Cube::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{
  if (m_minX > m_maxX) {
    std::cerr << "[ERROR] \t SourceGeometry_Cube::checkPosInfo: \n\t\t input minX is greater than input maxX! minX = \"" << m_minX
              << "\" maxX = \"" << m_maxX << "\"" << std::endl;
    exit(1);
  }
  if (m_minY > m_maxY) {
    std::cerr << "ERROR] \t SourceGeometry_Cube::checkPosInfo: \n\t\t input minY is greater than input maxY! minY = \"" << m_minY
              << "\" maxY = \"" << m_maxY << "\"" << std::endl;
    exit(1);
  }
  if (m_minZ > m_maxZ) {
    std::cerr << "ERROR] \t SourceGeometry_Cube::checkPosInfo: \n\t\t input minZ is greater than input maxZ! minZ = \"" << m_minZ
              << "\" maxZ = \"" << m_maxZ << "\"" << std::endl;
    exit(1);
  }

  if (m_minX < domainXstart) {
    std::cerr << "ERROR] \t SourceGeometry_Cube::checkPosInfo: \n\t\t input minX is outside of domain! minX = \"" << m_minX
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (m_minY < domainYstart) {
    std::cerr << "ERROR] \t SourceGeometry_Cube::checkPosInfo: \n\t\t input minY is outside of domain! minY = \"" << m_minY
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (m_minZ < domainZstart) {
    std::cerr << "ERROR] \t SourceGeometry_Cube::checkPosInfo: \n\t\t input minZ is outside of domain! minZ = \"" << m_minZ
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }

  if (m_maxX > domainXend) {
    std::cerr << "ERROR] \t SourceGeometry_Cube::checkPosInfo: \n\t\t input maxX is outside of domain! maxX = \"" << m_maxX
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (m_maxY > domainYend) {
    std::cerr << "ERROR] \t SourceGeometry_Cube::checkPosInfo: \n\t\t input maxY is outside of domain! maxY = \"" << m_maxY
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (m_maxZ > domainZend) {
    std::cerr << "ERROR] \t SourceGeometry_Cube::checkPosInfo: \n\t\t input maxZ is outside of domain! maxZ = \"" << m_maxZ
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}


void SourceGeometry_Cube::setInitialPosition(Particle *ptr)
{
  // generate uniform dist in domain
  ptr->xPos_init = uniformDistribution(prng) * (m_maxX - m_minX) + m_minX;
  ptr->yPos_init = uniformDistribution(prng) * (m_maxY - m_minY) + m_minY;
  ptr->zPos_init = uniformDistribution(prng) * (m_maxZ - m_minZ) + m_minZ;
}

void SourceGeometry_Cube::setInitialPosition(double &x, double &y, double &z)
{
  // generate uniform dist in domain
  x = uniformDistribution(prng) * (m_maxX - m_minX) + m_minX;
  y = uniformDistribution(prng) * (m_maxY - m_minY) + m_minY;
  z = uniformDistribution(prng) * (m_maxZ - m_minZ) + m_minZ;
}
