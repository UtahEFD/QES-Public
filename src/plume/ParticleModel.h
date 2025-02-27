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

#include <utility>

#include "util/DataSource.h"
#include "util/QESDataTransport.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "ManagedContainer.h"

#include "Particle.h"

#include "ParticleModel_Visitor.h"
#include "StatisticsDirector.h"
#include "Deposition.h"
#include "Source.h"

class PlumeInputData;
class PLUMEGeneralData;

class ParticleModel;
class ParticleModelBuilderInterface
{
public:
  /**
   * /brief
   */
  virtual ParticleModel *create(QESDataTransport &) = 0;
};

class ParticleModel
{
public:
  ParticleModel(QESDataTransport &, string);

  virtual ~ParticleModel();

  virtual void accept(ParticleModel_Visitor *visitor)
  {
    visitor->visit(this);
  }

  virtual void generateParticleList(QEStime &timeCurrent,
                                    const float &dt,
                                    WINDSGeneralData *WGD,
                                    TURBGeneralData *TGD,
                                    PLUMEGeneralData *PGD);

  virtual void advect(const float &timeRemainder,
                      WINDSGeneralData *WGD,
                      TURBGeneralData *TGD,
                      PLUMEGeneralData *PGD);

  virtual void process(QEStime &timeIn,
                       const float &dt,
                       WINDSGeneralData *WGD,
                       TURBGeneralData *TGD,
                       PLUMEGeneralData *PGD);

  int get_nbr_rogue() { return nbr_rogue; }
  std::string get_tag() { return tag; }

  virtual int get_nbr_active() { return (int)particles_control.get_nbr_active(); };
  virtual int get_nbr_inserted() { return (int)particles_control.get_nbr_inserted(); };

  void setStats(StatisticsDirector *in) { stats = in; }

  void addSource(Source *);
  void addSources(std::vector<Source *>);
  ParticleType getParticleType() { return particleType; }

  ManagedContainer<ParticleControl> particles_control;
  std::vector<ParticleCore> particles_core;
  std::vector<ParticleLSDM> particles_lsdm;
  std::vector<ParticleMetadata> particles_metadata;

  StatisticsDirector *stats = nullptr;
  Deposition *deposition = nullptr;

protected:
  explicit ParticleModel(ParticleType type, std::string tag_in)
    : particleType(type), tag(std::move(tag_in))
  {}

  ParticleType particleType{};
  std::string tag{};

  std::vector<Source *> sources;

  int nbr_rogue = 0;

private:
  ParticleModel() = default;
};

inline void ParticleModel::addSource(Source *newSource)
{
  sources.push_back(newSource);
}

inline void ParticleModel::addSources(std::vector<Source *> newSources)
{
  sources.insert(sources.end(), newSources.begin(), newSources.end());
}