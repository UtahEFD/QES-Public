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

/** @file SourceCircle.cpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "PI_SourceGeometry_SphereShell.hpp"

SourceComponent *PI_SourceGeometry_SphereShell::create(QESDataTransport &data)
{
  return new SourceGeometrySphereShell({ m_posX, m_posY, m_posZ }, radius);
}

/*
void PI_SourceGeometry_SphereShell::checkPosInfo(const float &domainXstart,
                                                 const float &domainXend,
                                                 const float &domainYstart,
                                                 const float &domainYend,
                                                 const float &domainZstart,
                                                 const float &domainZend)
{
  if (radius < 0) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_SphereShell::checkPosInfo: \n\t\t input radius is negative! radius = \"" << radius << "\"" << std::endl;
    exit(1);
  }

  if ((posX - radius) < domainXstart || (posX + radius) > domainXend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_SphereShell::checkPosInfo: \n\t\t input posX+radius is outside of domain! posX = \"" << posX << "\" radius = \"" << radius
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if ((posY - radius) < domainYstart || (posY + radius) > domainYend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_SphereShell::checkPosInfo: \n\t\t input posY+radius is outside of domain! posY = \"" << posY << "\" radius = \"" << radius
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if ((posZ - radius) < domainZstart || (posZ + radius) > domainZend) {
    std::cerr << "[ERROR] \t PI_SourceGeometry_SphereShell::checkPosInfo: \n\t\t input posZ is outside of domain! posZ = \"" << posZ << "\" radius = \"" << radius
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}

void PI_SourceGeometry_SphereShell::setInitialPosition(vec3 &p)
{
  // uniform distribution over surface of sphere
  float nx = normalDistribution(prng);
  float ny = normalDistribution(prng);
  float nz = normalDistribution(prng);
  float overn = 1 / sqrt(nx * nx + ny * ny + nz * nz);
  p._1 = posX + radius * nx * overn;
  p._2 = posY + radius * ny * overn;
  p._3 = posZ + radius * nz * overn;
}
*/