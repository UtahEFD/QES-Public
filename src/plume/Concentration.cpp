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

/** @file Concentration.cpp
 * @brief This class represents information stored for each particle
 */

#include "Concentration.h"
#include "PI_CollectionParameters.hpp"

Concentration::Concentration(const PI_CollectionParameters *colParams,
                             ManagedContainer<ParticleControl> &particles_control,
                             std::vector<ParticleCore> &particles_core)
  : averagingPeriod(colParams->averagingPeriod), ongoingAveragingTime(0.0),
    nBoxesX(colParams->nBoxesX), nBoxesY(colParams->nBoxesY), nBoxesZ(colParams->nBoxesZ),
    lBndx(colParams->boxBoundsX1), uBndx(colParams->boxBoundsX2),
    lBndy(colParams->boxBoundsY1), uBndy(colParams->boxBoundsY2),
    lBndz(colParams->boxBoundsZ1), uBndz(colParams->boxBoundsZ2),
    m_ptr_particles_control(&particles_control),
    m_ptr_particles_core(&particles_core)

{
  // setup output frequency control information
  // averagingStartTime = m_plume->getSimTimeStart() + PID->colParams->averagingStartTime;
  // averagingPeriod = colParams->averagingPeriod;

  // set the initial next output time value
  // nextOutputTime = averagingStartTime + averagingPeriod;

  // --------------------------------------------------------
  // setup information: sampling box/concentration
  // --------------------------------------------------------

  // Sampling box variables for calculating concentration data
  boxSizeX = (uBndx - lBndx) / ((float)nBoxesX);
  boxSizeY = (uBndy - lBndy) / ((float)nBoxesY);
  boxSizeZ = (uBndz - lBndz) / ((float)nBoxesZ);

  volume = boxSizeX * boxSizeY * boxSizeZ;

  // output concentration storage variables
  xBoxCen.resize(nBoxesX);
  yBoxCen.resize(nBoxesY);
  zBoxCen.resize(nBoxesZ);

  float zR = 0.0, yR = 0.0, xR = 0.0;
  for (int k = 0; k < nBoxesZ; ++k) {
    zBoxCen.at(k) = lBndz + (zR * boxSizeZ) + (boxSizeZ / 2.0f);
    zR = zR + 1.0f;
  }
  for (int j = 0; j < nBoxesY; ++j) {
    yBoxCen.at(j) = lBndy + (yR * boxSizeY) + (boxSizeY / 2.0f);
    yR = yR + 1.0f;
  }
  for (int i = 0; i < nBoxesX; ++i) {
    xBoxCen.at(i) = lBndx + (xR * boxSizeX) + (boxSizeX / 2.0f);
    xR = xR + 1.0f;
  }

  // initialization of the container
  pBox.resize(nBoxesX * nBoxesY * nBoxesZ, 0);
  conc.resize(nBoxesX * nBoxesY * nBoxesZ, 0.0);
}

void Concentration::collect(QEStime &timeIn, const float &timeStep)
{
  // for all particles see where they are relative to the concentration collection boxes
  for (int k = 0; k < m_ptr_particles_control->size(); ++k) {
    // because particles all start out as active now, need to also check the release time
    if (m_ptr_particles_control->at(k).state == ACTIVE) {
      ParticleCore *par = &m_ptr_particles_core->at(k);
      // Calculate which collection box this particle is currently in.
      // x-direction
      int idx = get_x_index(par->pos._1);
      // y-direction
      int idy = get_y_index(par->pos._2);
      // z-direction
      int idz = get_z_index(par->pos._3);

      // now, does the particle land in one of the boxes?
      // if so, add one particle to that box count
      if (idx >= 0 && idx <= nBoxesX - 1
          && idy >= 0 && idy <= nBoxesY - 1
          && idz >= 0 && idz <= nBoxesZ - 1) {
        int id = idz * nBoxesY * nBoxesX + idy * nBoxesX + idx;
        pBox[id]++;
        conc[id] = conc[id] + par->m * par->wdecay * timeStep;
      }

    }// is active == true

  }// particle loop
  ongoingAveragingTime += timeStep;
}

void Concentration::finalize(QEStime &timeIn)
{
  for (auto &c : conc) {
    c = c / (ongoingAveragingTime * volume);
  }
}
void Concentration::reset()
{
  // reset container for the next averaging period
  ongoingAveragingTime = 0.0;
  for (auto &p : pBox) {
    p = 0.0;
  }
  for (auto &c : conc) {
    c = 0.0;
  }
}

void Concentration::prepareDataAndPushToFile(QEStime t)
{
  finalize(t);
  pushToFile(t);
  reset();
}

void Concentration::setOutputFields()
{
  defineDimension("x_c", "x-center collection box", "m", &xBoxCen);
  defineDimension("y_c", "y-center collection box", "m", &yBoxCen);
  defineDimension("z_c", "z-center collection box", "m", &zBoxCen);

  defineDimensionSet("concentration", { "t", "z_c", "y_c", "x_c" });

  defineVariable("t_avg", "Averaging time", "s", "time", &ongoingAveragingTime);
  defineVariable("p_count", "number of particle per box", "#ofPar", "concentration", &pBox);
  defineVariable("c", "concentration", "g m-3", "concentration", &conc);
}
