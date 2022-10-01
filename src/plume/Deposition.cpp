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

/** @file Deposition.cpp
 * @brief
 */

#include "Deposition.h"

Deposition::Deposition(const WINDSGeneralData *WGD)
{
  numcell_cent = WGD->numcell_cent;

  x.resize(WGD->nx - 1);
  for (auto k = 0u; k < x.size(); ++k) {
    x[k] = WGD->x[k];
  }
  y.resize(WGD->ny - 1);
  for (auto k = 0u; k < y.size(); ++k) {
    y[k] = WGD->y[k];
  }
  z.resize(WGD->nz - 1);
  for (auto k = 0u; k < z.size(); ++k) {
    z[k] = WGD->z[k];
  }

  depcvol.resize(numcell_cent, 0.0);

  nbrFace = WGD->wall_below_indices.size()
            + WGD->wall_above_indices.size()
            + WGD->wall_back_indices.size()
            + WGD->wall_front_indices.size()
            + WGD->wall_left_indices.size()
            + WGD->wall_right_indices.size();
}
