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
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file LocalMixingDefault.cpp
 * @brief :document this:
 * @sa LocalMixing
 */

#include "LocalMixingDefault.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

void LocalMixingDefault::defineMixingLength(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();

  // seeding Local Mixing Length with the verical distance to the terrain (up to 2*max_z)
  for (int i = 0; i < nx - 1; i++) {
    for (int j = 0; j < ny - 1; j++) {
      for (int k = 1; k < nz - 2; k++) {
        int icell_cent = WGD->domain.cell(i, j, k);
        if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
          WGD->mixingLengths[icell_cent] = WGD->domain.z[k] - WGD->terrain[WGD->domain.cell2d(i, j)];
        }
        if (WGD->mixingLengths[icell_cent] < 0.0) {
          WGD->mixingLengths[icell_cent] = 0.0;
        }
      }
    }
  }
}
