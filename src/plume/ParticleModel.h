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

#include <utility>

#include "util/ManagedContainer.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "Deposition.h"
#include "ParticleContainers.h"
#include "Source.hpp"

class PlumeInputData;
class PLUMEGeneralData;

class TracerParticle_Model;
class HeavyParticle_Model;

class ModelVisitor
{
public:
  virtual void visitTracerParticle_Model(TracerParticle_Model *) = 0;
  virtual void visitHeavyParticle_Model(HeavyParticle_Model *) = 0;
};

class ParticleModel
{
public:
  virtual ~ParticleModel() = default;

  virtual void accept(ModelVisitor *visitor) = 0;

  virtual void initialize(const PlumeInputData *PID,
                          WINDSGeneralData *WGD,
                          TURBGeneralData *TGD,
                          PLUMEGeneralData *PGD) = 0;

  virtual void generateParticleList(QEStime &timeCurrent,
                                    const float &dt,
                                    WINDSGeneralData *WGD,
                                    TURBGeneralData *TGD,
                                    PLUMEGeneralData *PGD) = 0;
  virtual void advect(const double &timeRemainder,
                      WINDSGeneralData *WGD,
                      TURBGeneralData *TGD,
                      PLUMEGeneralData *PGD) = 0;
  virtual void process(QEStime &timeIn,
                       const float &dt,
                       WINDSGeneralData *WGD,
                       TURBGeneralData *TGD,
                       PLUMEGeneralData *PGD) = 0;

  int get_nbr_rogue() { return nbr_rogue; };
  virtual int get_nbr_active() = 0;
  virtual int get_nbr_inserted() = 0;


  ParticleType getParticleType()
  {
    return particleType;
  }

  std::string tag{};

protected:
  explicit ParticleModel(ParticleType type, std::string tag_in)
    : particleType(type), tag(std::move(tag_in))
  {}

  ParticleType particleType{};

  Deposition *deposition = nullptr;

  int nbr_rogue = 0;

private:
  ParticleModel() = default;
};
