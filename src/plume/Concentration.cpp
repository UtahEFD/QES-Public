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

/** @file Concentration.cpp
 * @brief
 */


#include "Concentration.h"

Concentration::Concentration(const CollectionParameters *colParams)
  : averagingPeriod(colParams->averagingPeriod), ongoingAveragingTime(0.0),
    nBoxesX(colParams->nBoxesX), nBoxesY(colParams->nBoxesY), nBoxesZ(colParams->nBoxesZ),
    lBndx(colParams->boxBoundsX1), uBndx(colParams->boxBoundsX2),
    lBndy(colParams->boxBoundsY1), uBndy(colParams->boxBoundsY2),
    lBndz(colParams->boxBoundsZ1), uBndz(colParams->boxBoundsZ2)
{
  // setup output frequency control information
  // averagingStartTime = m_plume->getSimTimeStart() + PID->colParams->averagingStartTime;
  averagingPeriod = colParams->averagingPeriod;

  // set the initial next output time value
  // nextOutputTime = averagingStartTime + averagingPeriod;

  // --------------------------------------------------------
  // setup information: sampling box/concentration
  // --------------------------------------------------------

  // Sampling box variables for calculating concentration data
  boxSizeX = (uBndx - lBndx) / (nBoxesX);
  boxSizeY = (uBndy - lBndy) / (nBoxesY);
  boxSizeZ = (uBndz - lBndz) / (nBoxesZ);

  volume = boxSizeX * boxSizeY * boxSizeZ;

  // output concentration storage variables
  xBoxCen.resize(nBoxesX);
  yBoxCen.resize(nBoxesY);
  zBoxCen.resize(nBoxesZ);

  int zR = 0, yR = 0, xR = 0;
  for (int k = 0; k < nBoxesZ; ++k) {
    zBoxCen.at(k) = lBndz + (zR * boxSizeZ) + (boxSizeZ / 2.0);
    zR++;
  }
  for (int j = 0; j < nBoxesY; ++j) {
    yBoxCen.at(j) = lBndy + (yR * boxSizeY) + (boxSizeY / 2.0);
    yR++;
  }
  for (int i = 0; i < nBoxesX; ++i) {
    xBoxCen.at(i) = lBndx + (xR * boxSizeX) + (boxSizeX / 2.0);
    xR++;
  }

  // initialization of the container
  pBox.resize(nBoxesX * nBoxesY * nBoxesZ, 0);
  conc.resize(nBoxesX * nBoxesY * nBoxesZ, 0.0);
}

void Concentration::reset()
{
  // reset container for the next averaging period
  ongoingAveragingTime = 0.0;
  for (auto p : pBox) {
    p = 0.0;
  }
  for (auto c : conc) {
    c = 0.0;
  }
}