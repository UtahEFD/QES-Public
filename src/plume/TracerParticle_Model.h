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

/** @file Particle.h
 * @brief This class represents information stored for each particle
 */

#pragma once

#include "util/ManagedContainer.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "Deposition.h"
#include "ParticleModel.h"

#include "TracerParticle.h"
#include "TracerParticle_Source.h"

class PI_TracerParticle;
class TracerParticle_Statistics;

class TracerParticle_Model : public ParticleModel
{
public:
  explicit TracerParticle_Model(const PI_TracerParticle *);

  ~TracerParticle_Model() = default;

  void accept(ModelVisitor *visitor) override
  {
    visitor->visitTracerParticle_Model(this);
  }

  void initialize(const PlumeInputData *PID,
                  WINDSGeneralData *WGD,
                  TURBGeneralData *TGD,
                  PLUMEGeneralData *PGD) override;

  void generateParticleList(QEStime &timeCurrent,
                            const float &dt,
                            WINDSGeneralData *WGD,
                            TURBGeneralData *TGD,
                            PLUMEGeneralData *PGD) override;

  void advect(const double &total_time_interval,
              WINDSGeneralData *WGD,
              TURBGeneralData *TGD,
              PLUMEGeneralData *PGD) override;

  void process(QEStime &timeIn,
               const float &dt,
               WINDSGeneralData *WGD,
               TURBGeneralData *TGD,
               PLUMEGeneralData *PGD) override;

  int get_nbr_active() override { return (int)particles->get_nbr_active(); };
  int get_nbr_inserted() override { return (int)particles->get_nbr_inserted(); };
  ManagedContainer<TracerParticle> *get_particles() { return particles; }

  void addSources(std::vector<TracerParticle_Source *>);

  // friend class declaration
  TracerParticle_Statistics *stats = nullptr;
  friend class TracerParticle_Statistics;

protected:
  ManagedContainer<TracerParticle> *particles{};
  std::vector<TracerParticle_Source *> sources;

private:
};