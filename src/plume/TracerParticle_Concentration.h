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

#pragma once

#include "util/ManagedContainer.h"
#include "util/DataSource.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "PLUMEGeneralData.h"
#include "TracerParticle.h"
#include "TracerParticle_Model.h"


class TracerParticle_Concentration : public StatisticsInterface
  , public DataSource
{
public:
  TracerParticle_Concentration(const PI_CollectionParameters *, TracerParticle_Model *);

  void collect(QEStime &timeIn, const float &timeStep) override;
  void finalize(QEStime &timeIn) override;
  void reset() override;

  void prepareDataAndPushToFile(QEStime) override;

protected:
  int get_x_index(const float &) const;
  int get_y_index(const float &) const;
  int get_z_index(const float &) const;

  void setOutputFields() override;

public:
  // averaging period in seconds
  float averagingPeriod;
  float ongoingAveragingTime;
  // Sampling box variables for calculating concentration data
  // Number of boxes to use for the sampling box
  int nBoxesX, nBoxesY, nBoxesZ;// Copies of the input: nBoxesX, Y, and Z.
  // upper & lower bounds in each direction of the sampling boxes
  float lBndx, lBndy, lBndz, uBndx, uBndy, uBndz;// Copies of the input: boxBoundsX1, boxBoundsX2, boxBoundsY1,
  float boxSizeX, boxSizeY, boxSizeZ;// these are the box sizes in each direction, calculated from nBoxes, lBnd, and uBnd variables
  float volume;// volume of the sampling boxes (=nBoxesX*nBoxesY*nBoxesZ)

  // output concentration storage variables
  std::vector<float> xBoxCen, yBoxCen, zBoxCen;// list of x,y, and z points for the concentration sampling box information
  std::vector<int> pBox;// sampling box particle counter (for average)
  std::vector<float> conc;// concentration values (for output)

  ManagedContainer<TracerParticle> *m_particles;

private:
  TracerParticle_Concentration() = default;
};


inline int TracerParticle_Concentration::get_x_index(const float &x) const
{
  return floor((x - lBndx) / (boxSizeX + 1e-9));
}
inline int TracerParticle_Concentration::get_y_index(const float &y) const
{
  return floor((y - lBndy) / (boxSizeY + 1e-9));
}
inline int TracerParticle_Concentration::get_z_index(const float &z) const
{
  return floor((z - lBndz) / (boxSizeZ + 1e-9));
}