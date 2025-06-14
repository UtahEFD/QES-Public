/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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

/** @file SourceComponent.h
 * @brief This class defines the interface for the source components.
 */

#pragma once

#include "util/QEStime.h"
#include "util/QESDataTransport.h"

class WINDSGeneralData;
class PLUMEGeneralData;

class SourceComponent;
class SourceComponentBuilderInterface
{
public:
  virtual SourceComponent *create(QESDataTransport &) = 0;
};

class SourceComponent
{
public:
  SourceComponent() = default;
  virtual ~SourceComponent() = default;
  virtual void generate(const QEStime &, const int &, QESDataTransport &) = 0;

protected:
};

class SetPhysicalProperties : public SourceComponent
{
public:
  SetPhysicalProperties(const float &particleDiameter, const float &particleDensity)
    : m_particleDiameter(particleDiameter), m_particleDensity(particleDensity)
  {}
  ~SetPhysicalProperties() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override
  {
    data.put("diameter", std::vector<float>(n, m_particleDiameter));
    data.put("density", std::vector<float>(n, m_particleDensity));
  }

private:
  SetPhysicalProperties() = default;

  float m_particleDiameter{};
  float m_particleDensity{};
};
