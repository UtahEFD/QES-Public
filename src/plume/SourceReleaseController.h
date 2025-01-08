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

/** @file SourceComponent.h
 * @brief This class defines the interface for the source components.
 */

#pragma once

#include "util/QEStime.h"
#include "util/QESDataTransport.h"

class SourceReleaseController;
class SourceReleaseControllerBuilderInterface
{
public:
  virtual SourceReleaseController *create(QESDataTransport &) = 0;
};

class SourceReleaseController
{
public:
  SourceReleaseController() = default;
  virtual ~SourceReleaseController() = default;

  virtual QEStime startTime() = 0;
  virtual QEStime endTime() = 0;
  virtual int particles(const QEStime &) = 0;
  virtual float mass(const QEStime &) = 0;

protected:
};

class SourceReleaseController_base : public SourceReleaseController
{
public:
  SourceReleaseController_base(const QEStime &s_time, const QEStime &e_time, const int &nbr_part, const float &total_mass)
    : m_startTime(s_time), m_endTime(e_time),
      m_particlePerTimestep(nbr_part), m_massPerTimestep(total_mass)
  {}
  ~SourceReleaseController_base() override = default;

  QEStime startTime() override { return m_startTime; }
  QEStime endTime() override { return m_endTime; }
  int particles(const QEStime &currTime) override { return m_particlePerTimestep; }
  float mass(const QEStime &currTime) override { return m_massPerTimestep; }

protected:
  QEStime m_startTime;
  QEStime m_endTime;
  int m_particlePerTimestep{};
  float m_massPerTimestep{};

private:
  SourceReleaseController_base() = default;
};