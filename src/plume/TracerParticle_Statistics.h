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

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "PLUMEGeneralData.h"
#include "TracerParticle.h"
#include "TracerParticle_Model.h"

#include "Concentration.hpp"


class TracerParticle_Concentration : public Concentration
{
public:
  explicit TracerParticle_Concentration(const PlumeInputData *PID, TracerParticle_Model *pm)
    : Concentration(PID->colParams)
  {
    m_particles = pm->get_particles();
  }
  void collect(QEStime &timeIn, const float &timeStep) override;
  void compute(QEStime &timeIn) override;

protected:
  ManagedContainer<TracerParticle> *m_particles;
};

class TracerParticle_Statistics
{
public:
  TracerParticle_Statistics(const PlumeInputData *PID, PLUMEGeneralData *PGD, TracerParticle_Model *pm)
  {
    averagingStartTime = PGD->getSimTimeStart() + PID->colParams->averagingStartTime;
    averagingPeriod = PID->colParams->averagingPeriod;
    nextOutputTime = averagingStartTime + averagingPeriod;
    concentration = new TracerParticle_Concentration(PID, pm);
  }

  void compute(QEStime &time,
               const float &dt,
               WINDSGeneralData *WGD,
               TURBGeneralData *TGD,
               PLUMEGeneralData *PGD,
               const TracerParticle_Model *);

  TracerParticle_Concentration *concentration;

  QEStime averagingStartTime;
  QEStime nextOutputTime;
  float averagingPeriod;
  float ongoingAveragingTime;

private:
  TracerParticle_Statistics() = default;
};
