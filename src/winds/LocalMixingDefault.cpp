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

  // z cell-center
  std::vector<float> z_cc;
  z_cc.resize(nz - 1, 0);
  z_cc = WGD->domain.z;

  // seeding Local Mixing Length with the verical distance to the terrain (up to 2*max_z)
  for (int i = 0; i < nx - 1; i++) {
    for (int j = 0; j < ny - 1; j++) {
      for (int k = 1; k < nz - 2; k++) {
        int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
          WGD->mixingLengths[icell_cent] = z_cc[k] - WGD->terrain[i + j * (nx - 1)];
        }
        if (WGD->mixingLengths[icell_cent] < 0.0) {
          WGD->mixingLengths[icell_cent] = 0.0;
        }
      }
    }
  }
}
