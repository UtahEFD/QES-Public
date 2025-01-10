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

/** @file SourceGeometryCube.h
 * @brief This class defines the interface for the source components.
 */

#pragma once

#include <random>

#include "util/QEStime.h"
#include "util/VectorMath.h"

#include "SourceComponent.h"
#include "SourceGeometryPoint.h"


class WINDSGeneralData;
class PLUMEGeneralData;

/**
 * \brief Source Geometry: Cube
 */
class SourceGeometryCube : public SourceComponent
{
public:
  SourceGeometryCube(const vec3 &min, const vec3 &max);
  ~SourceGeometryCube() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override;

private:
  SourceGeometryCube() = default;

  vec3 m_min{};
  vec3 m_max{};

  std::random_device rd;// Will be used to obtain a seed for the random number engine
  std::mt19937 prng;// Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> uniform;
};
