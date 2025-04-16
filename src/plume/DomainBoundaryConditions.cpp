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

/** @file DoaminBoundaryConditions.cpp
 * @brief
 */

#include "DomainBoundaryConditions.h"

void DomainBC_exiting::enforce(float &pos, float &velFluct, ParticleState &state)
{
  // if it goes out of the domain, set isActive to false
  if (pos <= domainStart || pos >= domainEnd) {
    state = INACTIVE;
  } else {
    state = ACTIVE;
  }
}

void DomainBC_periodic::enforce(float &pos, float &velFluct, ParticleState &state)
{

  float domainSize = domainEnd - domainStart;

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
  state = ACTIVE;
}

void DomainBC_reflection::enforce(float &pos, float &velFluct, ParticleState &state)
{

  int reflectCount = 0;
  while ((pos < domainStart || pos > domainEnd) && reflectCount < 100) {
    // past end of domain or before beginning of the domain
    if (pos > domainEnd) {
      pos = domainEnd - (pos - domainEnd);
      velFluct = -velFluct;
      // velFluct_old = -velFluct_old;
    } else if (pos < domainStart) {
      pos = domainStart - (pos - domainStart);
      velFluct = -velFluct;
      // velFluct_old = -velFluct_old;
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
      state = INACTIVE;
    } else if (pos < domainStart) {
      std::cout << "warning (Plume::enforceWallBCs_reflection): "
                << "lower boundary condition failed! Setting isActive to "
                   "false. xPos = \""
                << pos << "\"" << std::endl;
      state = INACTIVE;
    }
  }
  state = ACTIVE;
}
