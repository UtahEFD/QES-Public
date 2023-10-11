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

#include "SourceGeometry_SphereShell.hpp"
#include "winds/WINDSGeneralData.h"

void SourceGeometry_SphereShell::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{
  if (radius < 0) {
    std::cerr << "[ERROR] \t SourceGeometry_SphereShell::checkPosInfo: \n\t\t input radius is negative! radius = \"" << radius << "\"" << std::endl;
    exit(1);
  }

  if ((posX - radius) < domainXstart || (posX + radius) > domainXend) {
    std::cerr << "[ERROR] \t SourceGeometry_SphereShell::checkPosInfo: \n\t\t input posX+radius is outside of domain! posX = \"" << posX << "\" radius = \"" << radius
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if ((posY - radius) < domainYstart || (posY + radius) > domainYend) {
    std::cerr << "[ERROR] \t SourceGeometry_SphereShell::checkPosInfo: \n\t\t input posY+radius is outside of domain! posY = \"" << posY << "\" radius = \"" << radius
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if ((posZ - radius) < domainZstart || (posZ + radius) > domainZend) {
    std::cerr << "[ERROR] \t SourceGeometry_SphereShell::checkPosInfo: \n\t\t input posZ is outside of domain! posZ = \"" << posZ << "\" radius = \"" << radius
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}


void SourceGeometry_SphereShell::setInitialPosition(Particle *ptr)
{
  // uniform distribution over surface of sphere
  double nx = normalDistribution(prng);
  double ny = normalDistribution(prng);
  double nz = normalDistribution(prng);
  double overn = 1 / sqrt(nx * nx + ny * ny + nz * nz);
  ptr->xPos_init = posX + radius * nx * overn;
  ptr->yPos_init = posY + radius * ny * overn;
  ptr->zPos_init = posZ + radius * nz * overn;
}

void SourceGeometry_SphereShell::setInitialPosition(double &x, double &y, double &z)
{
  // uniform distribution over surface of sphere
  double nx = normalDistribution(prng);
  double ny = normalDistribution(prng);
  double nz = normalDistribution(prng);
  double overn = 1 / sqrt(nx * nx + ny * ny + nz * nz);
  x = posX + radius * nx * overn;
  y = posY + radius * ny * overn;
  z = posZ + radius * nz * overn;
}
