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

/** @file WallReflection.cpp 
 * @brief These functions handle the different wall reflection options
 *
 * @note Part of plume class 
 */

#include "WallReflection.h"
#include "Plume.hpp"

bool WallReflection_SetToInactive::reflect(const WINDSGeneralData *WGD,
                                           const Plume *plume,
                                           double &xPos,
                                           double &yPos,
                                           double &zPos,
                                           double &disX,
                                           double &disY,
                                           double &disZ,
                                           double &uFluct,
                                           double &vFluct,
                                           double &wFluct)

{
  try {
    int cellIdx = plume->interp->getCellId(xPos, yPos, zPos);
    int cellFlag(0);
    cellFlag = WGD->icellflag.at(cellIdx);

    if ((cellFlag == 0) || (cellFlag == 2)) {
      // particle end trajectory inside solide -> set inactive
      return false;
    } else {
      // particle end trajectory outside solide -> keep active
      return true;
    }

  } catch (const std::out_of_range &oor) {
    // cell ID out of bound (assuming particle outside of domain)
    //if (zPos < domainZstart) {
    //  std::cerr << "Reflection problem: particle out of range before reflection" << std::endl;
    //  std::cerr << xPos << "," << yPos << "," << zPos << std::endl;
    //}
    // -> set to false
    return false;
  }
}
