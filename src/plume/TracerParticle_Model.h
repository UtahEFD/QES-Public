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

/** @file Particle.h
 * @brief This class represents information stored for each particle
 */

#pragma once

#include "util/DataSource.h"
#include "util/QESDataTransport.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "Source.h"
#include "ParticleModel.h"
#include "ParticleModel_Visitor.h"

#include "TracerParticle.h"

#include "Deposition.h"

class TracerParticle_Concentration;

class TracerParticle_Model : public ParticleModel
{
public:
  TracerParticle_Model(QESDataTransport &, const string &);

  ~TracerParticle_Model();

  void accept(ParticleModel_Visitor *visitor) override
  {
    visitor->visit(this);
  }

  void generateParticleList(QEStime &timeCurrent,
                            const float &dt,
                            WINDSGeneralData *WGD,
                            TURBGeneralData *TGD,
                            PLUMEGeneralData *PGD) override;

  void advect(const float &total_time_interval,
              WINDSGeneralData *WGD,
              TURBGeneralData *TGD,
              PLUMEGeneralData *PGD) override;

  void process(QEStime &timeIn,
               const float &dt,
               WINDSGeneralData *WGD,
               TURBGeneralData *TGD,
               PLUMEGeneralData *PGD) override;

  int get_nbr_active() override { return (int)particles.get_nbr_active(); };
  int get_nbr_inserted() override { return (int)particles.get_nbr_inserted(); };
  ManagedContainer<TracerParticle> *get_particles() { return &particles; }

  // friend class declaration
  friend class TracerParticle_Concentration;

protected:
  ManagedContainer<TracerParticle> particles;

private:
};
