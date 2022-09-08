/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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

/** @file DoaminBoundaryConditions.cpp
 * @brief 
 */

#include "DomainBoundaryConditions.h"

bool DomainBC_exiting::enforce(double &pos, double &velFluct)
{
  // if it goes out of the domain, set isActive to false
  if (pos <= domainStart || pos >= domainEnd) {
    return false;
  } else {
    return true;
  }
}

bool DomainBC_periodic::enforce(double &pos, double &velFluct)
{

  double domainSize = domainEnd - domainStart;

  if (domainSize != 0) {
    // before beginning of the domain => add domain length
    while (pos < domainStart) {
      pos = pos + domainSize;
    }
    // past end of domain => sub domain length
    while (pos > domainEnd) {
      pos = pos - domainSize;
    }
  }

  return true;
}

bool DomainBC_reflection::enforce(double &pos, double &velFluct)
{

  int reflectCount = 0;
  while ((pos < domainStart || pos > domainEnd) && reflectCount < 100) {
    // past end of domain or before beginning of the domain
    if (pos > domainEnd) {
      pos = domainEnd - (pos - domainEnd);
      velFluct = -velFluct;
      //velFluct_old = -velFluct_old;
    } else if (pos < domainStart) {
      pos = domainStart - (pos - domainStart);
      velFluct = -velFluct;
      //velFluct_old = -velFluct_old;
    }
    reflectCount = reflectCount + 1;
  }// while outside of domain

  // if the velocity is so large that the particle would reflect more than 100
  // times, the boundary condition could fail.
  if (reflectCount == 100) {
    if (pos > domainEnd) {
      std::cout << "warning (Plume::enforceWallBCs_reflection): "
                << "upper boundary condition failed! Setting isActive to "
                   "false. pos = \""
                << pos << "\"" << std::endl;
      return false;
    } else if (pos < domainStart) {
      std::cout << "warning (Plume::enforceWallBCs_reflection): "
                << "lower boundary condition failed! Setting isActive to "
                   "false. xPos = \""
                << pos << "\"" << std::endl;
      return false;
    }
  }

  return true;
}
