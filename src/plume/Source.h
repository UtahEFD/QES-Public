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

/** @file SourceType.hpp
 * @brief  This class represents a generic sourece type
 *
 * @note Pure virtual child of ParseInterface
 * @sa ParseInterface
 */

#pragma once

#include <random>
#include <list>

#include "util/ParseInterface.h"

#include "Particle.h"
#include "ParticleIDGen.h"

#include "SourceReleaseController.h"
#include "SourceComponent.h"
#include "SourceIDGen.h"

class Source;
class SourceBuilderInterface
{
public:
  virtual Source *create(QESDataTransport &) = 0;
};


class SetParticleID : public SourceComponent
{
public:
  SetParticleID()
  {
    id_gen = ParticleIDGen::getInstance();
  }
  ~SetParticleID() override = default;
  void initialize(const WINDSGeneralData *WGD, const PLUMEGeneralData *PGD) override {}

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {
    std::vector<uint32_t> ids(n);
    id_gen->get(ids);
    data.put("ID", ids);
  }

private:
  ParticleIDGen *id_gen = nullptr;
};


class Source
{
private:
protected:
  int m_id;

  SourceReleaseController *m_release{};
  std::vector<SourceComponent *> m_components{};

  float total_mass = 0;
  int total_particle_released = 0;

public:
  QESDataTransport data;

  // constructor
  Source()
  {
    auto sourceID = SourceIDGen::getInstance();
    m_id = sourceID->get();

    m_components.emplace_back(new SetParticleID());
  }
  // constructor
  virtual ~Source()
  {
    delete m_release;
    for (auto c : m_components)
      delete c;
  }

  int getID() const { return m_id; }
  QESDataTransport &getData() { return data; }

  void addRelease(SourceReleaseController *r) { m_release = r; }
  void addComponent(SourceComponent *c) { m_components.emplace_back(c); }

  virtual bool isActive(const QEStime &currTime) const
  {
    return (currTime >= m_release->startTime() && currTime <= m_release->endTime());
  }

  virtual int generate(const QEStime &currTime)
  {
    // query how many particle need to be released
    if (isActive(currTime)) {
      // m_releaseType->m_particlePerTimestep;
      int n = m_release->particles(currTime);
      for (auto c : m_components)
        c->generate(currTime, n, data);

      // update source counter
      if (data.contains("mass")) {
        for (auto m : data.get_ref<std::vector<float>>("mass"))
          total_mass += m;
      }
      total_particle_released += n;

      return n;
    } else {
      data.clear();
      return 0;
    }
  }
  void print() const
  {
    std::cout << m_id << std::endl;
  }
};
