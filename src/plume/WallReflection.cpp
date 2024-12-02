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

/** @file WallReflection.cpp
 * @brief
 */

#include "WallReflection.h"

void WallReflection_SetToInactive::reflect(const WINDSGeneralData *WGD,
                                           vec3 &pos,
                                           vec3 &dist,
                                           vec3 &fluct,
                                           ParticleState &state)

{
  try {
    long cellIdx = m_interp->getCellId(pos);
    int cellFlag(0);
    cellFlag = WGD->icellflag.at(cellIdx);

    if ((cellFlag == 0) || (cellFlag == 2)) {
      // particle end trajectory inside solide -> set inactive
      state = INACTIVE;
    } else {
      // particle end trajectory outside solide -> keep active
      state = ACTIVE;
    }

  } catch (const std::out_of_range &oor) {
    // cell ID out of bound (assuming particle outside of domain)
    // if (zPos < domainZstart) {
    //  std::cerr << "Reflection problem: particle out of range before reflection" << std::endl;
    //  std::cerr << xPos << "," << yPos << "," << zPos << std::endl;
    //}
    // -> set to false
    state = INACTIVE;
  }
}
