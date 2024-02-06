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

/** @file TracerParticle_Statistics
 * @brief This class represents information stored for each particle
 */

#include "TracerParticle_Statistics.h"

void TracerParticle_Statistics::compute(QEStime &timeIn,
                                        const float &timeStep,
                                        WINDSGeneralData *WGD,
                                        TURBGeneralData *TGD,
                                        PLUMEGeneralData *PGD,
                                        const TracerParticle_Model *pm)
{
  // nextOutputTime = averagingStartTime + averagingPeriod;

  if (timeIn > averagingStartTime) {
    // incrementation of the averaging time
    ongoingAveragingTime += timeStep;
    concentration->collect(timeIn, timeStep);

    if (timeIn >= nextOutputTime) {
      // compute the stats
      concentration->compute(timeIn);

      // notify output

      // reset variables
      concentration->reset();

      nextOutputTime = nextOutputTime + averagingPeriod;
    }
  }
}

void TracerParticle_Concentration::collect(QEStime &timeIn, const float &timeStep)
{
  // for all particles see where they are relative to the concentration collection boxes
  for (auto &par : *m_particles) {
    // because particles all start out as active now, need to also check the release time
    if (par.isActive) {
      // Calculate which collection box this particle is currently in.
      // x-direction
      int idx = get_x_index(par.xPos);
      // y-direction
      int idy = get_y_index(par.yPos);
      // z-direction
      int idz = get_z_index(par.zPos);

      // now, does the particle land in one of the boxes?
      // if so, add one particle to that box count
      if (idx >= 0 && idx <= nBoxesX - 1
          && idy >= 0 && idy <= nBoxesY - 1
          && idz >= 0 && idz <= nBoxesZ - 1) {
        int id = idz * nBoxesY * nBoxesX + idy * nBoxesX + idx;
        pBox[id]++;
        conc[id] = conc[id] + par.m * par.wdecay * timeStep;
      }

    }// is active == true

  }// particle loop
}

void TracerParticle_Concentration::compute(QEStime &timeIn)
{
  for (auto c : conc) {
    c = c / (ongoingAveragingTime * volume);
  }
}
