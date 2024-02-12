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

/** @file SourceFullDomain.cpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "PI_SourceGeometry_FullDomain.hpp"

void PI_SourceGeometry_FullDomain::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{

  // notice that setting the variables as I am doing right now is not the standard way of doing this function
  xDomainStart = domainXstart;
  yDomainStart = domainYstart;
  zDomainStart = domainZstart;
  xDomainEnd = domainXend;
  yDomainEnd = domainYend;
  zDomainEnd = domainZend;


  if (xDomainStart > xDomainEnd) {
    std::cerr << "[ERROR] \t SourceGeometry_FullDomain::checkPosInfo: \n\t\t input xDomainStart is greater than input xDomainEnd! xDomainStart = \"" << xDomainStart
              << "\" xDomainEnd = \"" << xDomainEnd << "\"" << std::endl;
    exit(1);
  }
  if (yDomainStart > yDomainEnd) {
    std::cerr << "[ERROR] \t SourceGeometry_FullDomain::checkPosInfo: \n\t\t input yDomainStart is greater than input yDomainEnd! yDomainStart = \"" << yDomainStart
              << "\" yDomainEnd = \"" << yDomainEnd << "\"" << std::endl;
    exit(1);
  }
  if (zDomainStart > zDomainEnd) {
    std::cerr << "[ERROR] \t SourceGeometry_FullDomain::checkPosInfo: \n\t\t input zDomainStart is greater than input zDomainEnd! zDomainStart = \"" << zDomainStart
              << "\" zDomainEnd = \"" << zDomainEnd << "\"" << std::endl;
    exit(1);
  }

  // unfortunately there is no easy way to check that the input domain sizes are correct, so the code could potentially fail later on
  //  cause there is no easy checking method to be implemented here
}

void PI_SourceGeometry_FullDomain::setInitialPosition(double &x, double &y, double &z)
{
  // generate uniform dist in domain
  x = uniformDistribution(prng) * (xDomainEnd - xDomainStart) + xDomainStart;
  y = uniformDistribution(prng) * (yDomainEnd - yDomainStart) + yDomainStart;
  z = uniformDistribution(prng) * (zDomainEnd - zDomainStart) + zDomainStart;
}
