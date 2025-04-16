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

/** @file ReleaseType_continuous.hpp
 * @brief This class represents a specific release type.
 *
 * @note Child of ReleaseType
 * @sa ReleaseType
 */

#pragma once

#include "PI_ReleaseType.hpp"

class PI_ReleaseType_continuous : public PI_ReleaseType
{
private:
  float m_massPerSec = 0.0;

protected:
public:
  // Default constructor
  PI_ReleaseType_continuous() : PI_ReleaseType()
  {
  }

  // destructor
  ~PI_ReleaseType_continuous() override = default;

  void parseValues() override;
  void initialize(const float &timestep) override;
  SourceReleaseController *create(QESDataTransport &data) override;
};
